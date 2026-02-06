// flash_look_around_bwd_sm90.cuh - Backward kernel implementation for FA3-style look-around attention
//
// This implements the backward pass of look-around attention with:
// 1. Proper backward through normalization (D_i = dot(dO, O))
// 2. Correct transposed convolution: dS[j] = sum_k(w[k] * dP_conv[j - shift[k]])
// 3. Unified softmax backward after transposed convolution
// 4. Gradient accumulation for convolution weights
//
// The transposed 5-tap convolution formula for backward:
//   dP[j] = w0*dP_conv[j-2] + w1*dP_conv[j-1] + w2*dP_conv[j] + w3*dP_conv[j+1] + w4*dP_conv[j+2]

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>

#include "flash_look_around_kernel_sm90.h"

namespace flash_look_around {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward kernel implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL>
__global__ void flash_look_around_bwd_kernel_sm90(FlashLookAroundBwdParams params) {
    // Constants
    constexpr int WARP_SIZE = 32;
    constexpr int HALO_SIZE = 2;
    constexpr float NEG_INF_BWD = -1e30f;

    // Grid indices
    const int q_block_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    const int batch_idx = bh_idx / params.n_q_heads;
    const int q_head_idx = bh_idx % params.n_q_heads;

    // GQA: Map Q head to corresponding KV head
    const int heads_per_kv = params.n_q_heads / params.n_kv_heads;
    const int kv_head_idx = q_head_idx / heads_per_kv;

    // Thread indices
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = num_threads / WARP_SIZE;

    // Query range for this block
    const int q_start = q_block_idx * BLOCK_M;

    // Input/output pointers
    const __nv_bfloat16* Q_batch = reinterpret_cast<const __nv_bfloat16*>(params.Q_ptr)
        + batch_idx * params.Q_batch_stride + q_head_idx * params.Q_head_stride;
    const __nv_bfloat16* K_batch = reinterpret_cast<const __nv_bfloat16*>(params.K_ptr)
        + batch_idx * params.K_batch_stride + kv_head_idx * params.K_head_stride;
    const __nv_bfloat16* V_batch = reinterpret_cast<const __nv_bfloat16*>(params.V_ptr)
        + batch_idx * params.V_batch_stride + kv_head_idx * params.V_head_stride;
    const __nv_bfloat16* O_batch = reinterpret_cast<const __nv_bfloat16*>(params.O_ptr)
        + batch_idx * params.O_batch_stride + q_head_idx * params.O_head_stride;
    const float* LSE_batch = params.LSE_ptr
        + batch_idx * params.n_q_heads * params.T_q + q_head_idx * params.T_q;
    const __nv_bfloat16* dO_batch = reinterpret_cast<const __nv_bfloat16*>(params.dO_ptr)
        + batch_idx * params.dO_batch_stride + q_head_idx * params.dO_head_stride;
    const float* conv_weights = reinterpret_cast<const float*>(params.conv_weights_ptr)
        + kv_head_idx * 5;

    // Output gradient pointers
    __nv_bfloat16* dQ_batch = reinterpret_cast<__nv_bfloat16*>(params.dQ_ptr)
        + batch_idx * params.dQ_batch_stride + q_head_idx * params.dQ_head_stride;
    float* dK_batch = params.dK_ptr
        + batch_idx * params.dK_batch_stride + kv_head_idx * params.dK_head_stride;
    float* dV_batch = params.dV_ptr
        + batch_idx * params.dV_batch_stride + kv_head_idx * params.dV_head_stride;
    float* dConv_weights = params.dConv_weights_ptr + kv_head_idx * 5;

    const float sm_scale = params.sm_scale;
    const int T_q = params.T_q;
    const int T_k = params.T_k;
    const int D = params.D;
    const int window_left = params.window_left;

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

    // Load convolution weights
    __shared__ float w_shared[5];
    __shared__ float dw_accum[5];
    if (tid < 5) {
        w_shared[tid] = conv_weights[tid];
        dw_accum[tid] = 0.0f;
    }
    __syncthreads();

    const float w0 = w_shared[0];  // shift +2
    const float w1 = w_shared[1];  // shift +1
    const float w2 = w_shared[2];  // shift 0
    const float w3 = w_shared[3];  // shift -1
    const float w4 = w_shared[4];  // shift -2

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
            Q_smem[m * BLOCK_D + d] = __float2bfloat16(0.0f);
            dO_smem[m * BLOCK_D + d] = __float2bfloat16(0.0f);
            O_smem[m * BLOCK_D + d] = __float2bfloat16(0.0f);
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
                d_i += __bfloat162float(dO_smem[m * BLOCK_D + d]) *
                       __bfloat162float(O_smem[m * BLOCK_D + d]);
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
        int earliest_k = max(0, q_start - window_left - HALO_SIZE);
        k_iter_start = (earliest_k / BLOCK_N) * BLOCK_N;
    }

    if (IS_CAUSAL) {
        int latest_k = q_start + BLOCK_M - 1 + HALO_SIZE;
        k_iter_end = min(T_k, ((latest_k / BLOCK_N) + 1) * BLOCK_N);
    }

    // ============================================================
    // Main loop over K/V blocks
    // ============================================================
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
                K_smem[n * BLOCK_D + d] = __float2bfloat16(0.0f);
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
                V_smem[n * BLOCK_D + d] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // ============================================================
        // RECOMPUTE QK^T SCORES
        // ============================================================
        for (int i = tid; i < BLOCK_M * (BLOCK_N + 4); i += num_threads) {
            int m = i / (BLOCK_N + 4);
            int n_halo = i % (BLOCK_N + 4);
            int q_pos = q_start + m;
            int k_pos = k_block_start - 2 + n_halo;

            float sum = 0.0f;
            if (q_pos < T_q && k_pos >= 0 && k_pos < T_k) {
                #pragma unroll 8
                for (int d = 0; d < D; d++) {
                    sum += __bfloat162float(Q_smem[m * BLOCK_D + d]) *
                           __bfloat162float(K_smem[n_halo * BLOCK_D + d]);
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
        // COMPUTE dV
        // For each V position n, sum contributions from all query positions m
        // ============================================================
        for (int n = warp_id; n < BLOCK_N; n += num_warps) {
            int v_pos = k_block_start + n;
            if (v_pos >= T_k) continue;

            for (int d = lane_id; d < D; d += WARP_SIZE) {
                float dv_sum = 0.0f;

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
                            case 0: w = w4; break;  // k_pos - 2
                            case 1: w = w3; break;  // k_pos - 1
                            case 2: w = w2; break;  // k_pos
                            case 3: w = w1; break;  // k_pos + 1
                            case 4: w = w0; break;  // k_pos + 2
                        }
                        P_conv += w * p;
                    }

                    if (IS_CAUSAL && v_pos > q_pos) {
                        P_conv = 0.0f;
                    }

                    if (P_conv > 0.0f) {
                        float do_val = __bfloat162float(dO_smem[m * BLOCK_D + d]);
                        dv_sum += P_conv * do_val;
                    }
                }

                if (fabsf(dv_sum) > 1e-10f) {
                    atomicAdd(&dV_batch[v_pos * D + d], dv_sum);
                }
            }
        }
        __syncthreads();

        // ============================================================
        // COMPUTE dQ, dK, and weight gradients
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

            // 2. Compute dP_conv_grad = dO @ V^T - D_i (for each output position)
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
                    dp_raw += __bfloat162float(dO_smem[m * BLOCK_D + d]) *
                              __bfloat162float(V_smem[n * BLOCK_D + d]);
                }

                // Recompute P_conv for this position
                float P_conv = 0.0f;
                #pragma unroll
                for (int shift = 0; shift < 5; shift++) {
                    float w;
                    switch(shift) {
                        case 0: w = w4; break;
                        case 1: w = w3; break;
                        case 2: w = w2; break;
                        case 3: w = w1; break;
                        case 4: w = w0; break;
                    }
                    P_conv += w * P_halo[n + shift];
                }

                // dP_conv = (dO @ V^T - D_i) for the softmax backward
                // But we need P_conv for the full derivative: dP_conv * (1 - P_conv) would be wrong
                // Actually: d(P_conv @ V)/dP_conv = V^T, then backward through softmax
                dP_conv_grad[n] = dp_raw - d_i;
            }

            // 3. TRANSPOSED CONVOLUTION to get dP_halo
            // For dP[j] we need: sum over output positions k where P[j] contributes
            // Original: P_conv[k] = w0*P[k+2] + w1*P[k+1] + w2*P[k] + w3*P[k-1] + w4*P[k-2]
            // So P[j] contributes to:
            //   P_conv[j-2] with weight w0
            //   P_conv[j-1] with weight w1
            //   P_conv[j]   with weight w2
            //   P_conv[j+1] with weight w3
            //   P_conv[j+2] with weight w4
            // Thus: dP[j] = w0*dP_conv[j-2] + w1*dP_conv[j-1] + w2*dP_conv[j] + w3*dP_conv[j+1] + w4*dP_conv[j+2]
            #pragma unroll 4
            for (int n_halo = 0; n_halo < BLOCK_N + 4; n_halo++) {
                float dp = 0.0f;
                // Careful with indexing: n_halo is the halo index, we need to find which conv outputs it contributes to
                // n_halo corresponds to k_pos = k_block_start - 2 + n_halo
                // Output n corresponds to v_pos = k_block_start + n
                // So n_halo = n + 2 is the center position for output n
                // n_halo contributes to outputs [n_halo-4, n_halo] in the original indexing
                // In our output array (0 to BLOCK_N-1), that's [n_halo-4, n_halo] intersected with [0, BLOCK_N-1]
                if (n_halo >= 4 && n_halo - 4 < BLOCK_N) dp += w0 * dP_conv_grad[n_halo - 4];  // P[n_halo] -> P_conv[n_halo-2] with w0, but n_halo-2 = output + 2
                if (n_halo >= 3 && n_halo - 3 < BLOCK_N) dp += w1 * dP_conv_grad[n_halo - 3];
                if (n_halo >= 2 && n_halo - 2 < BLOCK_N) dp += w2 * dP_conv_grad[n_halo - 2];
                if (n_halo >= 1 && n_halo - 1 < BLOCK_N) dp += w3 * dP_conv_grad[n_halo - 1];
                if (n_halo < BLOCK_N)                    dp += w4 * dP_conv_grad[n_halo];
                dP_halo[n_halo] = dp;
            }

            // 4. Compute dS = P * dP * scale and accumulate dQ, dK
            #pragma unroll 4
            for (int n_halo = 0; n_halo < BLOCK_N + 4; n_halo++) {
                float dS_val = P_halo[n_halo] * dP_halo[n_halo] * sm_scale;
                if (fabsf(dS_val) < 1e-10f) continue;

                // Accumulate dQ
                #pragma unroll 8
                for (int d = 0; d < D; d++) {
                    dQ_smem[m * BLOCK_D + d] += dS_val * __bfloat162float(K_smem[n_halo * BLOCK_D + d]);
                }

                // Accumulate dK with atomics
                int k_pos = k_block_start - 2 + n_halo;
                if (k_pos >= 0 && k_pos < T_k) {
                    #pragma unroll 8
                    for (int d = 0; d < D; d++) {
                        atomicAdd(&dK_batch[k_pos * D + d],
                                  dS_val * __bfloat162float(Q_smem[m * BLOCK_D + d]));
                    }
                }
            }

            // 5. Accumulate weight gradients
            // dw[k] = sum over positions: dP_conv[j] * P[j + shift[k]]
            float dw0_local = 0.0f, dw1_local = 0.0f, dw2_local = 0.0f;
            float dw3_local = 0.0f, dw4_local = 0.0f;
            #pragma unroll 4
            for (int n = 0; n < BLOCK_N; n++) {
                if (fabsf(dP_conv_grad[n]) < 1e-10f) continue;
                // Forward: P_conv[n] = w0*P[n+4] + w1*P[n+3] + w2*P[n+2] + w3*P[n+1] + w4*P[n]
                // (in halo indexing where n+2 is the center)
                dw0_local += dP_conv_grad[n] * P_halo[n + 4];  // P[k+2] -> w0
                dw1_local += dP_conv_grad[n] * P_halo[n + 3];  // P[k+1] -> w1
                dw2_local += dP_conv_grad[n] * P_halo[n + 2];  // P[k]   -> w2
                dw3_local += dP_conv_grad[n] * P_halo[n + 1];  // P[k-1] -> w3
                dw4_local += dP_conv_grad[n] * P_halo[n];      // P[k-2] -> w4
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
            dQ_batch[q_pos * D + d] = __float2bfloat16(dQ_smem[m * BLOCK_D + d]);
        }
    }

    // Atomic add projection weight gradients
    __syncthreads();
    if (tid < 5) {
        atomicAdd(&dConv_weights[tid], dw_accum[tid]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel launcher implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

inline void launch_flash_look_around_bwd_sm90(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const __nv_bfloat16* O,
    const float* LSE,
    const float* conv_weights,
    const __nv_bfloat16* dO,
    __nv_bfloat16* dQ,
    float* dK,
    float* dV,
    float* dConv_weights,
    int B, int n_q_heads, int n_kv_heads, int T_q, int T_k, int D,
    float sm_scale,
    bool causal,
    int window_left,
    cudaStream_t stream
) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;

    // Prepare parameters
    FlashLookAroundBwdParams params;
    params.Q_ptr = Q;
    params.K_ptr = K;
    params.V_ptr = V;
    params.O_ptr = O;
    params.LSE_ptr = LSE;
    params.conv_weights_ptr = conv_weights;
    params.dO_ptr = dO;
    params.dQ_ptr = dQ;
    params.dK_ptr = dK;
    params.dV_ptr = dV;
    params.dConv_weights_ptr = dConv_weights;

    params.B = B;
    params.n_q_heads = n_q_heads;
    params.n_kv_heads = n_kv_heads;
    params.T_q = T_q;
    params.T_k = T_k;
    params.D = D;
    params.sm_scale = sm_scale;
    params.causal = causal;
    params.window_left = window_left;

    // Set up strides
    params.Q_batch_stride = n_q_heads * T_q * D;
    params.Q_head_stride = T_q * D;
    params.Q_seq_stride = D;
    params.K_batch_stride = n_kv_heads * T_k * D;
    params.K_head_stride = T_k * D;
    params.K_seq_stride = D;
    params.V_batch_stride = n_kv_heads * T_k * D;
    params.V_head_stride = T_k * D;
    params.V_seq_stride = D;
    params.O_batch_stride = n_q_heads * T_q * D;
    params.O_head_stride = T_q * D;
    params.O_seq_stride = D;
    params.dO_batch_stride = n_q_heads * T_q * D;
    params.dO_head_stride = T_q * D;
    params.dO_seq_stride = D;
    params.dQ_batch_stride = n_q_heads * T_q * D;
    params.dQ_head_stride = T_q * D;
    params.dQ_seq_stride = D;
    params.dK_batch_stride = n_kv_heads * T_k * D;
    params.dK_head_stride = T_k * D;
    params.dK_seq_stride = D;
    params.dV_batch_stride = n_kv_heads * T_k * D;
    params.dV_head_stride = T_k * D;
    params.dV_seq_stride = D;

    // Launch configuration
    int num_q_blocks = (T_q + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_q_blocks, B * n_q_heads);
    dim3 block(256);

    // Calculate shared memory size
    size_t smem_size = 0;
    if (D <= 64) {
        constexpr int BLOCK_D = 64;

        smem_size = BLOCK_M * BLOCK_D * sizeof(__nv_bfloat16) * 3 +  // Q, dO, O
                    BLOCK_M * sizeof(float) * 2 +                     // LSE, D_i
                    (BLOCK_N + 4) * BLOCK_D * sizeof(__nv_bfloat16) + // K with halos
                    BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16) +       // V
                    BLOCK_M * (BLOCK_N + 4) * sizeof(float) +         // S
                    BLOCK_M * BLOCK_D * sizeof(float) +               // dQ
                    5 * sizeof(float) * 2;                            // w_shared + dw_accum

        cudaFuncSetAttribute(
            causal ? flash_look_around_bwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, true>
                   : flash_look_around_bwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        if (causal) {
            flash_look_around_bwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, true>
                <<<grid, block, smem_size, stream>>>(params);
        } else {
            flash_look_around_bwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, false>
                <<<grid, block, smem_size, stream>>>(params);
        }
    } else {
        // D = 128 case - use smaller blocks to fit in shared memory
        constexpr int BLOCK_M_128 = 32;
        constexpr int BLOCK_N_128 = 32;
        constexpr int BLOCK_D = 128;

        int num_q_blocks_128 = (T_q + BLOCK_M_128 - 1) / BLOCK_M_128;
        dim3 grid_128(num_q_blocks_128, B * n_q_heads);
        dim3 block_128(128);

        smem_size = BLOCK_M_128 * BLOCK_D * sizeof(__nv_bfloat16) * 3 +
                    BLOCK_M_128 * sizeof(float) * 2 +
                    (BLOCK_N_128 + 4) * BLOCK_D * sizeof(__nv_bfloat16) +
                    BLOCK_N_128 * BLOCK_D * sizeof(__nv_bfloat16) +
                    BLOCK_M_128 * (BLOCK_N_128 + 4) * sizeof(float) +
                    BLOCK_M_128 * BLOCK_D * sizeof(float) +
                    5 * sizeof(float) * 2;

        cudaFuncSetAttribute(
            causal ? flash_look_around_bwd_kernel_sm90<BLOCK_M_128, BLOCK_N_128, BLOCK_D, true>
                   : flash_look_around_bwd_kernel_sm90<BLOCK_M_128, BLOCK_N_128, BLOCK_D, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        if (causal) {
            flash_look_around_bwd_kernel_sm90<BLOCK_M_128, BLOCK_N_128, BLOCK_D, true>
                <<<grid_128, block_128, smem_size, stream>>>(params);
        } else {
            flash_look_around_bwd_kernel_sm90<BLOCK_M_128, BLOCK_N_128, BLOCK_D, false>
                <<<grid_128, block_128, smem_size, stream>>>(params);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash_look_around
