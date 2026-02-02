// fused_look_around_bwd.cuh - Backward kernel for fused look-around flash attention
// Fixes the bugs in the Triton implementation:
// 1. Proper backward through normalization (D_i term)
// 2. Correct transposed convolution with index shifting
// 3. Unified softmax backward after transposed convolution
//
// PERFORMANCE FIX: Parallelized over m (each thread handles different query rows)

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

// Backward kernel for fused look-around flash attention
// PARALLELIZED: Each thread handles rows m = tid, tid + num_threads, ...
template <int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL>
__global__ void fused_look_around_flash_bwd_kernel(
    // Inputs from forward
    const __nv_bfloat16* __restrict__ Q,  // (B, H, T_q, D)
    const __nv_bfloat16* __restrict__ K,  // (B, H, T_k, D)
    const __nv_bfloat16* __restrict__ V,  // (B, H, T_k, D)
    const __nv_bfloat16* __restrict__ O,  // (B, H, T_q, D)
    const float* __restrict__ LSE,        // (B, H, T_q) log-sum-exp from forward
    const float* __restrict__ proj_weights,  // (H, 5) pre-softmaxed
    const __nv_bfloat16* __restrict__ dO,    // (B, H, T_q, D) gradient of output
    // Outputs
    __nv_bfloat16* __restrict__ dQ,       // (B, H, T_q, D)
    float* __restrict__ dK,               // (B, H, T_k, D) - float for atomic add
    float* __restrict__ dV,               // (B, H, T_k, D) - float for atomic add
    float* __restrict__ dProj_weights,    // (H, 5)
    int B, int H, int T_q, int T_k, int D,
    float sm_scale
) {
    // Grid: (num_q_blocks, B * H)
    const int q_block_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    const int batch_idx = bh_idx / H;
    const int head_idx = bh_idx % H;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Query range for this block
    const int q_start = q_block_idx * BLOCK_M;

    // Base pointers
    const __nv_bfloat16* Q_batch = Q + (batch_idx * H + head_idx) * T_q * D;
    const __nv_bfloat16* K_batch = K + (batch_idx * H + head_idx) * T_k * D;
    const __nv_bfloat16* V_batch = V + (batch_idx * H + head_idx) * T_k * D;
    const __nv_bfloat16* O_batch = O + (batch_idx * H + head_idx) * T_q * D;
    const float* LSE_batch = LSE + (batch_idx * H + head_idx) * T_q;
    const __nv_bfloat16* dO_batch = dO + (batch_idx * H + head_idx) * T_q * D;
    __nv_bfloat16* dQ_batch = dQ + (batch_idx * H + head_idx) * T_q * D;
    float* dK_batch = dK + (batch_idx * H + head_idx) * T_k * D;
    float* dV_batch = dV + (batch_idx * H + head_idx) * T_k * D;

    // Load projection weights (broadcast to all threads via shared memory)
    __shared__ float w_shared[5];
    if (tid < 5) {
        w_shared[tid] = proj_weights[head_idx * 5 + tid];
    }
    __syncthreads();

    float w0 = w_shared[0];  // shift +2
    float w1 = w_shared[1];  // shift +1
    float w2 = w_shared[2];  // shift 0
    float w3 = w_shared[3];  // shift -1
    float w4 = w_shared[4];  // shift -2

    // Shared memory layout
    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* dO_smem = Q_smem + BLOCK_M * BLOCK_D;
    __nv_bfloat16* O_smem = dO_smem + BLOCK_M * BLOCK_D;
    float* LSE_smem = reinterpret_cast<float*>(O_smem + BLOCK_M * BLOCK_D);
    float* D_i_smem = LSE_smem + BLOCK_M;  // D_i = dot(dO[i], O[i])
    __nv_bfloat16* K_smem = reinterpret_cast<__nv_bfloat16*>(D_i_smem + BLOCK_M);
    __nv_bfloat16* V_smem = K_smem + (BLOCK_N + 4) * BLOCK_D;
    float* S_smem = reinterpret_cast<float*>(V_smem + BLOCK_N * BLOCK_D);
    float* dQ_smem = S_smem + BLOCK_M * (BLOCK_N + 4);

    // Gradient accumulators for projection weights (per block, use shared memory)
    __shared__ float dw_accum[5];
    if (tid < 5) {
        dw_accum[tid] = 0.0f;
    }
    __syncthreads();

    // Load Q, dO, O, LSE to shared memory (parallelized)
    for (int i = tid; i < BLOCK_M * BLOCK_D; i += num_threads) {
        int m = i / BLOCK_D;
        int d = i % BLOCK_D;
        int q_pos = q_start + m;
        if (q_pos < T_q && d < D) {
            Q_smem[m * BLOCK_D + d] = Q_batch[q_pos * D + d];
            dO_smem[m * BLOCK_D + d] = dO_batch[q_pos * D + d];
            O_smem[m * BLOCK_D + d] = O_batch[q_pos * D + d];
        } else {
            Q_smem[m * BLOCK_D + d] = bf16_to_float_bwd(0.0f);
            dO_smem[m * BLOCK_D + d] = bf16_to_float_bwd(0.0f);
            O_smem[m * BLOCK_D + d] = bf16_to_float_bwd(0.0f);
        }
    }
    for (int i = tid; i < BLOCK_M; i += num_threads) {
        int q_pos = q_start + i;
        if (q_pos < T_q) {
            LSE_smem[i] = LSE_batch[q_pos];
        } else {
            LSE_smem[i] = 0.0f;
        }
    }
    __syncthreads();

    // Compute D_i = dot(dO[i], O[i]) - PARALLELIZED over m
    for (int m = tid; m < BLOCK_M; m += num_threads) {
        int q_pos = q_start + m;
        if (q_pos < T_q) {
            float d_i = 0.0f;
            for (int d = 0; d < D; d++) {
                float do_val = bf16_to_float_bwd(dO_smem[m * BLOCK_D + d]);
                float o_val = bf16_to_float_bwd(O_smem[m * BLOCK_D + d]);
                d_i += do_val * o_val;
            }
            D_i_smem[m] = d_i;
        } else {
            D_i_smem[m] = 0.0f;
        }
    }

    // Initialize dQ accumulator (will be owned by threads based on m assignment)
    for (int i = tid; i < BLOCK_M * D; i += num_threads) {
        dQ_smem[i] = 0.0f;
    }
    __syncthreads();

    // Iterate over K/V blocks
    for (int k_block_start = 0; k_block_start < T_k; k_block_start += BLOCK_N) {
        // Load K with halos - PARALLELIZED
        for (int i = tid; i < (BLOCK_N + 4) * BLOCK_D; i += num_threads) {
            int n = i / BLOCK_D;
            int d = i % BLOCK_D;
            int k_pos = k_block_start - 2 + n;
            if (k_pos >= 0 && k_pos < T_k && d < D) {
                K_smem[n * BLOCK_D + d] = K_batch[k_pos * D + d];
            } else {
                K_smem[n * BLOCK_D + d] = bf16_to_float_bwd(0.0f);
            }
        }

        // Load V - PARALLELIZED
        for (int i = tid; i < BLOCK_N * BLOCK_D; i += num_threads) {
            int n = i / BLOCK_D;
            int d = i % BLOCK_D;
            int v_pos = k_block_start + n;
            if (v_pos < T_k && d < D) {
                V_smem[n * BLOCK_D + d] = V_batch[v_pos * D + d];
            } else {
                V_smem[n * BLOCK_D + d] = bf16_to_float_bwd(0.0f);
            }
        }
        __syncthreads();

        // Compute S_halo = Q @ K^T (with halos) - PARALLELIZED
        for (int i = tid; i < BLOCK_M * (BLOCK_N + 4); i += num_threads) {
            int m = i / (BLOCK_N + 4);
            int n_halo = i % (BLOCK_N + 4);
            int q_pos = q_start + m;
            int k_pos = k_block_start - 2 + n_halo;

            if (q_pos < T_q && k_pos >= 0 && k_pos < T_k) {
                float sum = 0.0f;
                for (int d = 0; d < D; d++) {
                    float q_val = bf16_to_float_bwd(Q_smem[m * BLOCK_D + d]);
                    float k_val = bf16_to_float_bwd(K_smem[n_halo * BLOCK_D + d]);
                    sum += q_val * k_val;
                }
                sum *= sm_scale;

                if (IS_CAUSAL && k_pos > q_pos) {
                    sum = NEG_INF_BWD;
                }
                S_smem[m * (BLOCK_N + 4) + n_halo] = sum;
            } else {
                S_smem[m * (BLOCK_N + 4) + n_halo] = NEG_INF_BWD;
            }
        }
        __syncthreads();

        // ==========================================================
        // MAIN BACKWARD COMPUTATION - PARALLELIZED OVER M
        // Each thread handles rows m = tid, tid + num_threads, ...
        // ==========================================================
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            int q_pos = q_start + m;
            if (q_pos >= T_q) continue;

            float lse_i = LSE_smem[m];
            float d_i = D_i_smem[m];

            // Thread-local arrays for this row
            float P_halo[BLOCK_N + 4];
            float P_conv[BLOCK_N];
            float dP_conv_grad[BLOCK_N];
            float dP_halo[BLOCK_N + 4];

            // 1. Recompute P_halo = exp(S - lse_i)
            for (int n = 0; n < BLOCK_N + 4; n++) {
                float s = S_smem[m * (BLOCK_N + 4) + n];
                P_halo[n] = (s > -1e20f) ? expf(s - lse_i) : 0.0f;
            }

            // 2. Forward convolution: P_halo -> P_conv
            for (int n = 0; n < BLOCK_N; n++) {
                int k_pos = k_block_start + n;
                if (k_pos >= T_k || (IS_CAUSAL && k_pos > q_pos)) {
                    P_conv[n] = 0.0f;
                    continue;
                }
                // P_conv[n] = w0*P[n+2] + w1*P[n+1] + w2*P[n] + w3*P[n-1] + w4*P[n-2]
                // Index mapping: P[k+2] -> P_halo[n+4], P[k+1] -> P_halo[n+3], etc.
                float p_p2 = P_halo[n + 4];
                float p_p1 = P_halo[n + 3];
                float p_0  = P_halo[n + 2];
                float p_m1 = P_halo[n + 1];
                float p_m2 = P_halo[n];

                P_conv[n] = w0 * p_p2 + w1 * p_p1 + w2 * p_0 + w3 * p_m1 + w4 * p_m2;
            }

            // 3. Accumulate dV = P_conv^T @ dO (atomic to global)
            for (int n = 0; n < BLOCK_N; n++) {
                int v_pos = k_block_start + n;
                if (v_pos >= T_k || (IS_CAUSAL && v_pos > q_pos)) continue;
                if (fabsf(P_conv[n]) < 1e-9f) continue;

                for (int d = 0; d < D; d++) {
                    float do_val = bf16_to_float_bwd(dO_smem[m * BLOCK_D + d]);
                    atomicAdd(&dV_batch[v_pos * D + d], P_conv[n] * do_val);
                }
            }

            // 4. Compute dP_conv_raw = dO @ V^T, then apply normalization backward
            //    dP_conv_grad = P_conv * (dP_conv_raw - D_i)
            for (int n = 0; n < BLOCK_N; n++) {
                int v_pos = k_block_start + n;
                if (v_pos >= T_k || (IS_CAUSAL && v_pos > q_pos)) {
                    dP_conv_grad[n] = 0.0f;
                    continue;
                }

                float dp_raw = 0.0f;
                for (int d = 0; d < D; d++) {
                    float do_val = bf16_to_float_bwd(dO_smem[m * BLOCK_D + d]);
                    float v_val = bf16_to_float_bwd(V_smem[n * BLOCK_D + d]);
                    dp_raw += do_val * v_val;
                }

                // FIX #1: Proper backward through normalization
                dP_conv_grad[n] = P_conv[n] * (dp_raw - d_i);
            }

            // 5. Transposed convolution: dP_conv_grad -> dP_halo
            // FIX #2: Correct index shifting for transposed convolution
            for (int n_halo = 0; n_halo < BLOCK_N + 4; n_halo++) {
                float dp = 0.0f;
                // Reversed convolution: if P_halo[n_halo] contributed to P_conv[j] via weight w_tap,
                // then dP_conv_grad[j] contributes to dP_halo[n_halo] via the same w_tap
                // P_halo[n_halo] contributed to:
                //   P_conv[n_halo-4] via w0 (shift +2)
                //   P_conv[n_halo-3] via w1 (shift +1)
                //   P_conv[n_halo-2] via w2 (shift 0)
                //   P_conv[n_halo-1] via w3 (shift -1)
                //   P_conv[n_halo] via w4 (shift -2)
                if (n_halo >= 4 && n_halo - 4 < BLOCK_N) dp += w0 * dP_conv_grad[n_halo - 4];
                if (n_halo >= 3 && n_halo - 3 < BLOCK_N) dp += w1 * dP_conv_grad[n_halo - 3];
                if (n_halo >= 2 && n_halo - 2 < BLOCK_N) dp += w2 * dP_conv_grad[n_halo - 2];
                if (n_halo >= 1 && n_halo - 1 < BLOCK_N) dp += w3 * dP_conv_grad[n_halo - 1];
                if (n_halo < BLOCK_N)                      dp += w4 * dP_conv_grad[n_halo];

                dP_halo[n_halo] = dp;
            }

            // 6. Unified softmax backward
            // FIX #3: Single softmax backward after combining all dP contributions
            // dS = P * (dP - sum(P * dP))
            float row_sum_p_dp = 0.0f;
            for (int n = 0; n < BLOCK_N + 4; n++) {
                row_sum_p_dp += P_halo[n] * dP_halo[n];
            }

            // 7. Compute dS and accumulate dQ and dK
            for (int n_halo = 0; n_halo < BLOCK_N + 4; n_halo++) {
                float dS_val = P_halo[n_halo] * (dP_halo[n_halo] - row_sum_p_dp);
                dS_val *= sm_scale;

                if (fabsf(dS_val) < 1e-10f) continue;

                // Accumulate dQ (thread owns row m, no collision)
                for (int d = 0; d < D; d++) {
                    float k_val = bf16_to_float_bwd(K_smem[n_halo * BLOCK_D + d]);
                    dQ_smem[m * BLOCK_D + d] += dS_val * k_val;
                }

                // Accumulate dK (atomic to global)
                int k_pos = k_block_start - 2 + n_halo;
                if (k_pos >= 0 && k_pos < T_k) {
                    for (int d = 0; d < D; d++) {
                        float q_val = bf16_to_float_bwd(Q_smem[m * BLOCK_D + d]);
                        atomicAdd(&dK_batch[k_pos * D + d], dS_val * q_val);
                    }
                }
            }

            // 8. Accumulate weight gradients
            // dw[tap] = sum(dP_conv_grad * P_shifted)
            float dw0_local = 0.0f, dw1_local = 0.0f, dw2_local = 0.0f;
            float dw3_local = 0.0f, dw4_local = 0.0f;
            for (int n = 0; n < BLOCK_N; n++) {
                if (fabsf(dP_conv_grad[n]) < 1e-10f) continue;
                dw0_local += dP_conv_grad[n] * P_halo[n + 4];
                dw1_local += dP_conv_grad[n] * P_halo[n + 3];
                dw2_local += dP_conv_grad[n] * P_halo[n + 2];
                dw3_local += dP_conv_grad[n] * P_halo[n + 1];
                dw4_local += dP_conv_grad[n] * P_halo[n];
            }

            // Atomic add to shared accumulator
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

    // Atomic add projection weight gradients to global accumulator
    __syncthreads();
    if (tid < 5) {
        atomicAdd(&dProj_weights[head_idx * 5 + tid], dw_accum[tid]);
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
    int B, int H, int T_q, int T_k, int D,
    float sm_scale,
    bool causal,
    cudaStream_t stream
) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;

    int num_q_blocks = (T_q + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(128);

    // Shared memory size
    size_t smem_size = 0;

    // Get device properties for shared memory configuration
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    if (D <= 64) {
        // Q_smem + dO_smem + O_smem + LSE_smem + D_i_smem + K_smem + V_smem + S_smem + dQ_smem + dw_accum
        smem_size = BLOCK_M * 64 * 2 * 3 +       // Q, dO, O (bf16)
                    BLOCK_M * 4 * 2 +             // LSE, D_i (float)
                    (BLOCK_N + 4) * 64 * 2 +     // K with halos (bf16)
                    BLOCK_N * 64 * 2 +           // V (bf16)
                    BLOCK_M * (BLOCK_N + 4) * 4 + // S (float)
                    BLOCK_M * 64 * 4 +           // dQ (float)
                    5 * 4 +                       // w_shared
                    5 * 4;                        // dw_accum

        // Set max dynamic shared memory if needed
        if (smem_size > props.sharedMemPerBlock) {
            cudaFuncSetAttribute(
                causal ? fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 64, true>
                       : fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 64, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size
            );
        }

        if (causal) {
            fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 64, true>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, O, LSE, proj_weights, dO,
                    dQ, dK, dV, dProj_weights,
                    B, H, T_q, T_k, D, sm_scale
                );
        } else {
            fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 64, false>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, O, LSE, proj_weights, dO,
                    dQ, dK, dV, dProj_weights,
                    B, H, T_q, T_k, D, sm_scale
                );
        }
    } else {
        smem_size = BLOCK_M * 128 * 2 * 3 +
                    BLOCK_M * 4 * 2 +
                    (BLOCK_N + 4) * 128 * 2 +
                    BLOCK_N * 128 * 2 +
                    BLOCK_M * (BLOCK_N + 4) * 4 +
                    BLOCK_M * 128 * 4 +
                    5 * 4 +
                    5 * 4;

        if (causal) {
            fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 128, true>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, O, LSE, proj_weights, dO,
                    dQ, dK, dV, dProj_weights,
                    B, H, T_q, T_k, D, sm_scale
                );
        } else {
            fused_look_around_flash_bwd_kernel<BLOCK_M, BLOCK_N, 128, false>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, O, LSE, proj_weights, dO,
                    dQ, dK, dV, dProj_weights,
                    B, H, T_q, T_k, D, sm_scale
                );
        }
    }
}

}  // namespace fused_look_around
