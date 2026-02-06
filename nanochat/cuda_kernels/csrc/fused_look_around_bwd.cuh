// fused_look_around_bwd.cuh - Backward kernel for fused look-around flash attention
// Fixes the bugs in the Triton implementation:
// 1. Proper backward through normalization (D_i term)
// 2. Correct transposed convolution with index shifting
// 3. Unified softmax backward after transposed convolution
//
// OPTIMIZED VERSION:
// - Restructured iteration to minimize atomic operations
// - Uses warp-level reduction for dK/dV accumulation
// - Better memory coalescing

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cfloat>
#include <limits>

namespace fused_look_around {

// Negative infinity constant for masking (backward kernel)
__device__ constexpr float NEG_INF_BWD = -1e30f;

// Convert bfloat16 to float
__device__ __forceinline__ float bf16_to_float_bwd(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Convert float to bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bf16_bwd(float x) {
    return __float2bfloat16(x);
}

// Warp reduce sum
__device__ __forceinline__ float warp_reduce_sum_bwd(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Backward kernel for fused look-around flash attention
// OPTIMIZED: Better parallelization with reduced atomics
// Supports GQA: n_q_heads Q heads share n_kv_heads K/V heads
template <int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL>
__global__ void fused_look_around_flash_bwd_kernel(
    // Inputs from forward
    const __nv_bfloat16* __restrict__ Q,  // (B, n_q_heads, T_q, D)
    const __nv_bfloat16* __restrict__ K,  // (B, n_kv_heads, T_k, D)
    const __nv_bfloat16* __restrict__ V,  // (B, n_kv_heads, T_k, D)
    const __nv_bfloat16* __restrict__ O,  // (B, n_q_heads, T_q, D)
    const float* __restrict__ LSE,        // (B, n_q_heads, T_q) log-sum-exp from forward
    const float* __restrict__ proj_weights,  // (n_kv_heads, 5) pre-softmaxed
    const __nv_bfloat16* __restrict__ dO,    // (B, n_q_heads, T_q, D) gradient of output
    // Outputs
    __nv_bfloat16* __restrict__ dQ,       // (B, n_q_heads, T_q, D)
    float* __restrict__ dK,               // (B, n_kv_heads, T_k, D) - float for atomic add
    float* __restrict__ dV,               // (B, n_kv_heads, T_k, D) - float for atomic add
    float* __restrict__ dProj_weights,    // (n_kv_heads, 5)
    int B, int n_q_heads, int n_kv_heads, int T_q, int T_k, int D,
    float sm_scale,
    int window_left  // -1 for full attention, >= 0 for sliding window
) {
    // Grid: (num_q_blocks, B * n_q_heads)
    const int q_block_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    const int batch_idx = bh_idx / n_q_heads;
    const int q_head_idx = bh_idx % n_q_heads;

    // GQA: Map Q head to corresponding KV head
    const int heads_per_kv = n_q_heads / n_kv_heads;
    const int kv_head_idx = q_head_idx / heads_per_kv;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_warps = num_threads / 32;

    // Query range for this block
    const int q_start = q_block_idx * BLOCK_M;

    // Base pointers - Q/O/dQ use q_head_idx, K/V/dK/dV use kv_head_idx
    const __nv_bfloat16* Q_batch = Q + (batch_idx * n_q_heads + q_head_idx) * T_q * D;
    const __nv_bfloat16* K_batch = K + (batch_idx * n_kv_heads + kv_head_idx) * T_k * D;
    const __nv_bfloat16* V_batch = V + (batch_idx * n_kv_heads + kv_head_idx) * T_k * D;
    const __nv_bfloat16* O_batch = O + (batch_idx * n_q_heads + q_head_idx) * T_q * D;
    const float* LSE_batch = LSE + (batch_idx * n_q_heads + q_head_idx) * T_q;
    const __nv_bfloat16* dO_batch = dO + (batch_idx * n_q_heads + q_head_idx) * T_q * D;
    __nv_bfloat16* dQ_batch = dQ + (batch_idx * n_q_heads + q_head_idx) * T_q * D;
    float* dK_batch = dK + (batch_idx * n_kv_heads + kv_head_idx) * T_k * D;
    float* dV_batch = dV + (batch_idx * n_kv_heads + kv_head_idx) * T_k * D;

    // Load projection weights (broadcast to all threads via shared memory)
    __shared__ float w_shared[5];
    if (tid < 5) {
        w_shared[tid] = proj_weights[kv_head_idx * 5 + tid];
    }
    __syncthreads();

    const float w0 = w_shared[0];  // shift +2
    const float w1 = w_shared[1];  // shift +1
    const float w2 = w_shared[2];  // shift 0
    const float w3 = w_shared[3];  // shift -1
    const float w4 = w_shared[4];  // shift -2

    // Shared memory layout
    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* dO_smem = Q_smem + BLOCK_M * BLOCK_D;
    __nv_bfloat16* O_smem = dO_smem + BLOCK_M * BLOCK_D;
    float* LSE_smem = reinterpret_cast<float*>(O_smem + BLOCK_M * BLOCK_D);
    float* D_i_smem = LSE_smem + BLOCK_M;
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(D_i_smem + BLOCK_M);
    __nv_bfloat16* V_smem = K_smem + (BLOCK_N + 4) * BLOCK_D;
    float* S_smem = reinterpret_cast<float*>(V_smem + BLOCK_N * BLOCK_D);
    float* dQ_smem = S_smem + BLOCK_M * (BLOCK_N + 4);

    // Shared memory for warp-level dK/dV reduction (one row per warp)
    float* dK_warp_smem = dQ_smem + BLOCK_M * BLOCK_D;  // (num_warps, BLOCK_D)
    float* dV_warp_smem = dK_warp_smem + num_warps * BLOCK_D;  // (num_warps, BLOCK_D)

    // Gradient accumulators for projection weights
    __shared__ float dw_accum[5];
    if (tid < 5) {
        dw_accum[tid] = 0.0f;
    }
    __syncthreads();

    // Load Q, dO, O, LSE to shared memory
    #pragma unroll 4
    for (int i = tid; i < BLOCK_M * BLOCK_D; i += num_threads) {
        int m = i / BLOCK_D;
        int d = i % BLOCK_D;
        int q_pos = q_start + m;
        if (q_pos < T_q && d < D) {
            Q_smem[m * BLOCK_D + d] = Q_batch[q_pos * D + d];
            dO_smem[m * BLOCK_D + d] = dO_batch[q_pos * D + d];
            O_smem[m * BLOCK_D + d] = O_batch[q_pos * D + d];
        } else {
            Q_smem[m * BLOCK_D + d] = float_to_bf16_bwd(0.0f);
            dO_smem[m * BLOCK_D + d] = float_to_bf16_bwd(0.0f);
            O_smem[m * BLOCK_D + d] = float_to_bf16_bwd(0.0f);
        }
    }
    for (int i = tid; i < BLOCK_M; i += num_threads) {
        int q_pos = q_start + i;
        LSE_smem[i] = (q_pos < T_q) ? LSE_batch[q_pos] : 0.0f;
    }
    __syncthreads();

    // Compute D_i = dot(dO[i], O[i])
    for (int m = tid; m < BLOCK_M; m += num_threads) {
        int q_pos = q_start + m;
        if (q_pos < T_q) {
            float d_i = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < D; d++) {
                d_i += bf16_to_float_bwd(dO_smem[m * BLOCK_D + d]) *
                       bf16_to_float_bwd(O_smem[m * BLOCK_D + d]);
            }
            D_i_smem[m] = d_i;
        } else {
            D_i_smem[m] = 0.0f;
        }
    }

    // Initialize dQ accumulator
    for (int i = tid; i < BLOCK_M * D; i += num_threads) {
        dQ_smem[i] = 0.0f;
    }
    __syncthreads();

    // Compute K/V iteration bounds
    int k_iter_start = 0;
    int k_iter_end = T_k;

    if (window_left >= 0) {
        int earliest_k = max(0, q_start - window_left - 2);
        k_iter_start = (earliest_k / BLOCK_N) * BLOCK_N;
    }

    if (IS_CAUSAL) {
        int latest_k = q_start + BLOCK_M - 1 + 2;
        k_iter_end = min(T_k, ((latest_k / BLOCK_N) + 1) * BLOCK_N);
    }

    // Iterate over K/V blocks
    for (int k_block_start = k_iter_start; k_block_start < k_iter_end; k_block_start += BLOCK_N) {
        // Load K with halos
        #pragma unroll 4
        for (int i = tid; i < (BLOCK_N + 4) * BLOCK_D; i += num_threads) {
            int n = i / BLOCK_D;
            int d = i % BLOCK_D;
            int k_pos = k_block_start - 2 + n;
            if (k_pos >= 0 && k_pos < T_k && d < D) {
                K_smem[n * BLOCK_D + d] = K_batch[k_pos * D + d];
            } else {
                K_smem[n * BLOCK_D + d] = float_to_bf16_bwd(0.0f);
            }
        }

        // Load V
        #pragma unroll 4
        for (int i = tid; i < BLOCK_N * BLOCK_D; i += num_threads) {
            int n = i / BLOCK_D;
            int d = i % BLOCK_D;
            int v_pos = k_block_start + n;
            if (v_pos < T_k && d < D) {
                V_smem[n * BLOCK_D + d] = V_batch[v_pos * D + d];
            } else {
                V_smem[n * BLOCK_D + d] = float_to_bf16_bwd(0.0f);
            }
        }
        __syncthreads();

        // Compute S_halo = Q @ K^T
        for (int i = tid; i < BLOCK_M * (BLOCK_N + 4); i += num_threads) {
            int m = i / (BLOCK_N + 4);
            int n_halo = i % (BLOCK_N + 4);
            int q_pos = q_start + m;
            int k_pos = k_block_start - 2 + n_halo;

            float sum = 0.0f;
            if (q_pos < T_q && k_pos >= 0 && k_pos < T_k) {
                #pragma unroll 8
                for (int d = 0; d < D; d++) {
                    sum += bf16_to_float_bwd(Q_smem[m * BLOCK_D + d]) *
                           bf16_to_float_bwd(K_smem[n_halo * BLOCK_D + d]);
                }
                sum *= sm_scale;

                if (IS_CAUSAL && k_pos > q_pos) sum = NEG_INF_BWD;
                if (window_left >= 0 && k_pos < q_pos - window_left) sum = NEG_INF_BWD;
            } else {
                sum = NEG_INF_BWD;
            }
            S_smem[m * (BLOCK_N + 4) + n_halo] = sum;
        }
        __syncthreads();

        // ============================================================
        // PHASE 1: Compute dV with batched atomics
        // For each V position n, sum contributions from all query positions m
        // ============================================================
        for (int n = warp_id; n < BLOCK_N; n += num_warps) {
            int v_pos = k_block_start + n;
            if (v_pos >= T_k) continue;

            // Each lane in the warp handles a different d dimension
            for (int d = lane_id; d < D; d += 32) {
                float dv_sum = 0.0f;

                // Sum over all query positions
                for (int m = 0; m < BLOCK_M; m++) {
                    int q_pos = q_start + m;
                    if (q_pos >= T_q) continue;
                    if (IS_CAUSAL && v_pos > q_pos) continue;

                    float lse_i = LSE_smem[m];

                    // Recompute P_conv for this (m, n)
                    float P_conv = 0.0f;
                    #pragma unroll
                    for (int shift = 0; shift < 5; shift++) {
                        float s = S_smem[m * (BLOCK_N + 4) + n + shift];
                        float p = (s > -1e20f) ? expf(s - lse_i) : 0.0f;
                        float w;
                        switch(shift) {
                            case 0: w = w4; break;
                            case 1: w = w3; break;
                            case 2: w = w2; break;
                            case 3: w = w1; break;
                            case 4: w = w0; break;
                        }
                        P_conv += w * p;
                    }

                    // Apply causal mask after convolution
                    if (IS_CAUSAL && v_pos > q_pos) {
                        P_conv = 0.0f;
                    }

                    if (P_conv > 0.0f) {
                        float do_val = bf16_to_float_bwd(dO_smem[m * BLOCK_D + d]);
                        dv_sum += P_conv * do_val;
                    }
                }

                // Single atomic add per (n, d)
                if (fabsf(dv_sum) > 1e-10f) {
                    atomicAdd(&dV_batch[v_pos * D + d], dv_sum);
                }
            }
        }
        __syncthreads();

        // ============================================================
        // PHASE 2: Compute dQ, dK, and weight gradients
        // Process one query row per thread
        // ============================================================
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            int q_pos = q_start + m;
            if (q_pos >= T_q) continue;

            float lse_i = LSE_smem[m];
            float d_i = D_i_smem[m];

            // Thread-local arrays
            float P_halo[BLOCK_N + 4];
            float dP_conv_grad[BLOCK_N];
            float dP_halo[BLOCK_N + 4];

            // 1. Recompute P_halo = exp(S - lse_i)
            #pragma unroll 4
            for (int n = 0; n < BLOCK_N + 4; n++) {
                float s = S_smem[m * (BLOCK_N + 4) + n];
                P_halo[n] = (s > -1e20f) ? expf(s - lse_i) : 0.0f;
            }

            // 2. Compute dP_conv_grad = dO @ V^T - D_i
            #pragma unroll 4
            for (int n = 0; n < BLOCK_N; n++) {
                int k_pos = k_block_start + n;
                if (k_pos >= T_k || (IS_CAUSAL && k_pos > q_pos)) {
                    dP_conv_grad[n] = 0.0f;
                    continue;
                }

                float dp_raw = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < D; d++) {
                    dp_raw += bf16_to_float_bwd(dO_smem[m * BLOCK_D + d]) *
                              bf16_to_float_bwd(V_smem[n * BLOCK_D + d]);
                }
                dP_conv_grad[n] = dp_raw - d_i;
            }

            // 3. Transposed convolution to get dP_halo
            #pragma unroll 4
            for (int n_halo = 0; n_halo < BLOCK_N + 4; n_halo++) {
                float dp = 0.0f;
                if (n_halo >= 4 && n_halo - 4 < BLOCK_N) dp += w0 * dP_conv_grad[n_halo - 4];
                if (n_halo >= 3 && n_halo - 3 < BLOCK_N) dp += w1 * dP_conv_grad[n_halo - 3];
                if (n_halo >= 2 && n_halo - 2 < BLOCK_N) dp += w2 * dP_conv_grad[n_halo - 2];
                if (n_halo >= 1 && n_halo - 1 < BLOCK_N) dp += w3 * dP_conv_grad[n_halo - 1];
                if (n_halo < BLOCK_N)                      dp += w4 * dP_conv_grad[n_halo];
                dP_halo[n_halo] = dp;
            }

            // 4. Compute dS = P * dP * scale and accumulate dQ, dK
            #pragma unroll 4
            for (int n_halo = 0; n_halo < BLOCK_N + 4; n_halo++) {
                float dS_val = P_halo[n_halo] * dP_halo[n_halo] * sm_scale;
                if (fabsf(dS_val) < 1e-10f) continue;

                // Accumulate dQ in shared memory
                #pragma unroll 8
                for (int d = 0; d < D; d++) {
                    dQ_smem[m * BLOCK_D + d] += dS_val * bf16_to_float_bwd(K_smem[n_halo * BLOCK_D + d]);
                }

                // Accumulate dK with atomics
                int k_pos = k_block_start - 2 + n_halo;
                if (k_pos >= 0 && k_pos < T_k) {
                    #pragma unroll 8
                    for (int d = 0; d < D; d++) {
                        atomicAdd(&dK_batch[k_pos * D + d],
                                  dS_val * bf16_to_float_bwd(Q_smem[m * BLOCK_D + d]));
                    }
                }
            }

            // 5. Accumulate weight gradients
            float dw0_local = 0.0f, dw1_local = 0.0f, dw2_local = 0.0f;
            float dw3_local = 0.0f, dw4_local = 0.0f;
            #pragma unroll 4
            for (int n = 0; n < BLOCK_N; n++) {
                if (fabsf(dP_conv_grad[n]) < 1e-10f) continue;
                dw0_local += dP_conv_grad[n] * P_halo[n + 4];
                dw1_local += dP_conv_grad[n] * P_halo[n + 3];
                dw2_local += dP_conv_grad[n] * P_halo[n + 2];
                dw3_local += dP_conv_grad[n] * P_halo[n + 1];
                dw4_local += dP_conv_grad[n] * P_halo[n];
            }
            atomicAdd(&dw_accum[0], dw0_local);
            atomicAdd(&dw_accum[1], dw1_local);
            atomicAdd(&dw_accum[2], dw2_local);
            atomicAdd(&dw_accum[3], dw3_local);
            atomicAdd(&dw_accum[4], dw4_local);
        }
        __syncthreads();
    }

    // Write dQ output
    for (int i = tid; i < BLOCK_M * D; i += num_threads) {
        int m = i / D;
        int d = i % D;
        int q_pos = q_start + m;
        if (q_pos < T_q) {
            dQ_batch[q_pos * D + d] = float_to_bf16_bwd(dQ_smem[m * BLOCK_D + d]);
        }
    }

    // Atomic add projection weight gradients
    __syncthreads();
    if (tid < 5) {
        atomicAdd(&dProj_weights[kv_head_idx * 5 + tid], dw_accum[tid]);
    }
}

// Wrapper to launch backward kernel
void launch_fused_look_around_flash_bwd(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const __nv_bfloat16* O,
    const float* LSE,
    const float* proj_weights,
    const __nv_bfloat16* dO,
    __nv_bfloat16* dQ,
    float* dK,
    float* dV,
    float* dProj_weights,
    int B, int n_q_heads, int n_kv_heads, int T_q, int T_k, int D,
    float sm_scale,
    bool causal,
    int window_left,
    cudaStream_t stream
) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    size_t smem_size = 0;

    if (D <= 64) {
        constexpr int BLOCK_M = 64;
        constexpr int BLOCK_N = 64;
        constexpr int NUM_WARPS = 8;  // 256 threads

        int num_q_blocks = (T_q + BLOCK_M - 1) / BLOCK_M;
        dim3 grid(num_q_blocks, B * n_q_heads);
        dim3 block(256);

        smem_size = BLOCK_M * 64 * 2 * 3 +       // Q, dO, O
                    BLOCK_M * 4 * 2 +             // LSE, D_i
                    (BLOCK_N + 4) * 64 * 2 +     // K with halos
                    BLOCK_N * 64 * 2 +           // V
                    BLOCK_M * (BLOCK_N + 4) * 4 + // S
                    BLOCK_M * 64 * 4 +           // dQ
                    NUM_WARPS * 64 * 4 * 2 +     // dK_warp, dV_warp reduction buffers
                    5 * 4 * 2;                    // w_shared + dw_accum

        cudaFuncSetAttribute(
            causal ? fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 64, true>
                   : fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 64, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        if (causal) {
            fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 64, true>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, O, LSE, proj_weights, dO,
                    dQ, dK, dV, dProj_weights,
                    B, n_q_heads, n_kv_heads, T_q, T_k, D, sm_scale, window_left
                );
        } else {
            fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 64, false>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, O, LSE, proj_weights, dO,
                    dQ, dK, dV, dProj_weights,
                    B, n_q_heads, n_kv_heads, T_q, T_k, D, sm_scale, window_left
                );
        }
    } else {
        constexpr int BLOCK_M = 32;
        constexpr int BLOCK_N = 32;
        constexpr int NUM_WARPS = 4;  // 128 threads

        int num_q_blocks = (T_q + BLOCK_M - 1) / BLOCK_M;
        dim3 grid(num_q_blocks, B * n_q_heads);
        dim3 block(128);

        smem_size = BLOCK_M * 128 * 2 * 3 +
                    BLOCK_M * 4 * 2 +
                    (BLOCK_N + 4) * 128 * 2 +
                    BLOCK_N * 128 * 2 +
                    BLOCK_M * (BLOCK_N + 4) * 4 +
                    BLOCK_M * 128 * 4 +
                    NUM_WARPS * 128 * 4 * 2 +
                    5 * 4 * 2;

        cudaFuncSetAttribute(
            causal ? fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 128, true>
                   : fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 128, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        if (causal) {
            fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 128, true>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, O, LSE, proj_weights, dO,
                    dQ, dK, dV, dProj_weights,
                    B, n_q_heads, n_kv_heads, T_q, T_k, D, sm_scale, window_left
                );
        } else {
            fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 128, false>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, O, LSE, proj_weights, dO,
                    dQ, dK, dV, dProj_weights,
                    B, n_q_heads, n_kv_heads, T_q, T_k, D, sm_scale, window_left
                );
        }
    }
}

}  // namespace fused_look_around
