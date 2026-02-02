// fused_look_around_fwd.cuh - Forward kernel for fused look-around flash attention
// Implements 5-tap convolution on attention scores with online softmax
//
// PERFORMANCE FIX: Parallelized over m (each thread handles different query rows)

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cfloat>
#include <limits>

namespace fused_look_around {

// Kernel configuration
constexpr int HALO_SIZE = 2;  // +-2 for 5-tap convolution

// Negative infinity constant for masking
__device__ constexpr float NEG_INF = -1e30f;

// Convert bfloat16 to float
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Convert float to bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max_fwd(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum_fwd(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Forward kernel for fused look-around flash attention
// PARALLELIZED: Each thread handles rows m = tid, tid + num_threads, ...
template <int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL>
__global__ void fused_look_around_flash_fwd_kernel(
    const __nv_bfloat16* __restrict__ Q,  // (B, H, T_q, D)
    const __nv_bfloat16* __restrict__ K,  // (B, H, T_k, D)
    const __nv_bfloat16* __restrict__ V,  // (B, H, T_k, D)
    const float* __restrict__ proj_weights,  // (H, 5) pre-softmaxed weights
    __nv_bfloat16* __restrict__ O,        // (B, H, T_q, D)
    float* __restrict__ LSE,              // (B, H, T_q) log-sum-exp for backward
    int B, int H, int T_q, int T_k, int D,
    float sm_scale
) {
    // Grid: (num_q_blocks, B * H)
    const int q_block_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    const int batch_idx = bh_idx / H;
    const int head_idx = bh_idx % H;

    // Thread indexing
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Query range for this block
    const int q_start = q_block_idx * BLOCK_M;

    // Base pointers
    const __nv_bfloat16* Q_batch = Q + (batch_idx * H + head_idx) * T_q * D;
    const __nv_bfloat16* K_batch = K + (batch_idx * H + head_idx) * T_k * D;
    const __nv_bfloat16* V_batch = V + (batch_idx * H + head_idx) * T_k * D;
    __nv_bfloat16* O_batch = O + (batch_idx * H + head_idx) * T_q * D;
    float* LSE_batch = LSE + (batch_idx * H + head_idx) * T_q;

    // Load projection weights (broadcast via shared memory)
    __shared__ float w_shared[5];
    if (tid < 5) {
        w_shared[tid] = proj_weights[head_idx * 5 + tid];
    }
    __syncthreads();

    float w0 = w_shared[0];  // Weight for shift +2
    float w1 = w_shared[1];  // Weight for shift +1
    float w2 = w_shared[2];  // Weight for shift 0 (center)
    float w3 = w_shared[3];  // Weight for shift -1
    float w4 = w_shared[4];  // Weight for shift -2

    // Shared memory layout:
    // - Q_smem: (BLOCK_M, BLOCK_D) for query block
    // - K_smem: (BLOCK_N + 4, BLOCK_D) for key block with halos
    // - V_smem: (BLOCK_N, BLOCK_D) for value block
    // - S_smem: (BLOCK_M, BLOCK_N + 4) for QK scores with halos
    // - m_smem, l_smem, acc_smem: per-query accumulators
    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* K_smem = Q_smem + BLOCK_M * BLOCK_D;
    __nv_bfloat16* V_smem = K_smem + (BLOCK_N + 4) * BLOCK_D;
    float* S_smem = reinterpret_cast<float*>(V_smem + BLOCK_N * BLOCK_D);
    float* m_smem = S_smem + BLOCK_M * (BLOCK_N + 4);  // Running max per query
    float* l_smem = m_smem + BLOCK_M;                  // Running sum per query
    float* acc_smem = l_smem + BLOCK_M;                // Output accumulator (BLOCK_M, D)

    // Load Q block to shared memory - PARALLELIZED
    for (int i = tid; i < BLOCK_M * BLOCK_D; i += num_threads) {
        int m = i / BLOCK_D;
        int d = i % BLOCK_D;
        int q_pos = q_start + m;
        if (q_pos < T_q && d < D) {
            Q_smem[m * BLOCK_D + d] = Q_batch[q_pos * D + d];
        } else {
            Q_smem[m * BLOCK_D + d] = float_to_bf16(0.0f);
        }
    }

    // Initialize per-query accumulators - PARALLELIZED
    for (int m = tid; m < BLOCK_M; m += num_threads) {
        m_smem[m] = NEG_INF;
        l_smem[m] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M * D; i += num_threads) {
        acc_smem[i] = 0.0f;
    }
    __syncthreads();

    // Iterate over K/V blocks
    for (int k_block_start = 0; k_block_start < T_k; k_block_start += BLOCK_N) {
        // Load K with halos: positions [k_block_start - 2, k_block_start + BLOCK_N + 2)
        for (int i = tid; i < (BLOCK_N + 4) * BLOCK_D; i += num_threads) {
            int n = i / BLOCK_D;
            int d = i % BLOCK_D;
            int k_pos = k_block_start - 2 + n;  // Offset by -2 for halo
            if (k_pos >= 0 && k_pos < T_k && d < D) {
                K_smem[n * BLOCK_D + d] = K_batch[k_pos * D + d];
            } else {
                K_smem[n * BLOCK_D + d] = float_to_bf16(0.0f);
            }
        }

        // Load V: positions [k_block_start, k_block_start + BLOCK_N)
        for (int i = tid; i < BLOCK_N * BLOCK_D; i += num_threads) {
            int n = i / BLOCK_D;
            int d = i % BLOCK_D;
            int v_pos = k_block_start + n;
            if (v_pos < T_k && d < D) {
                V_smem[n * BLOCK_D + d] = V_batch[v_pos * D + d];
            } else {
                V_smem[n * BLOCK_D + d] = float_to_bf16(0.0f);
            }
        }
        __syncthreads();

        // Compute QK^T for all halo positions - PARALLELIZED
        for (int i = tid; i < BLOCK_M * (BLOCK_N + 4); i += num_threads) {
            int m = i / (BLOCK_N + 4);
            int n_halo = i % (BLOCK_N + 4);
            int q_pos = q_start + m;
            int k_pos = k_block_start - 2 + n_halo;

            if (q_pos < T_q && k_pos >= 0 && k_pos < T_k) {
                float sum = 0.0f;
                for (int d = 0; d < D; d++) {
                    float q_val = bf16_to_float(Q_smem[m * BLOCK_D + d]);
                    float k_val = bf16_to_float(K_smem[n_halo * BLOCK_D + d]);
                    sum += q_val * k_val;
                }
                sum *= sm_scale;

                // Apply causal mask
                if (IS_CAUSAL && k_pos > q_pos) {
                    sum = NEG_INF;
                }

                S_smem[m * (BLOCK_N + 4) + n_halo] = sum;
            } else {
                S_smem[m * (BLOCK_N + 4) + n_halo] = NEG_INF;
            }
        }
        __syncthreads();

        // ==========================================================
        // MAIN FORWARD COMPUTATION - PARALLELIZED OVER M
        // Each thread handles rows m = tid, tid + num_threads, ...
        // ==========================================================
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            int q_pos = q_start + m;
            if (q_pos >= T_q) continue;

            float m_i = m_smem[m];
            float l_i = l_smem[m];

            // Find max over all 5 shifted QK arrays for this K block
            // NOTE: Causal masking is already applied to S_smem during QK computation,
            // so we don't need to check it here again. The shifted scores for invalid
            // K positions will already be NEG_INF.
            float m_ij = NEG_INF;
            for (int n = 0; n < BLOCK_N; n++) {
                int v_pos = k_block_start + n;
                if (v_pos >= T_k) continue;

                // Get QK scores for all 5 shifts
                // Each score already has causal mask applied (NEG_INF for invalid K positions)
                float s_m2 = S_smem[m * (BLOCK_N + 4) + n];      // shift -2: uses K[j-2]
                float s_m1 = S_smem[m * (BLOCK_N + 4) + n + 1];  // shift -1: uses K[j-1]
                float s_0  = S_smem[m * (BLOCK_N + 4) + n + 2];  // shift 0: uses K[j]
                float s_p1 = S_smem[m * (BLOCK_N + 4) + n + 3];  // shift +1: uses K[j+1]
                float s_p2 = S_smem[m * (BLOCK_N + 4) + n + 4];  // shift +2: uses K[j+2]

                m_ij = fmaxf(m_ij, s_m2);
                m_ij = fmaxf(m_ij, s_m1);
                m_ij = fmaxf(m_ij, s_0);
                m_ij = fmaxf(m_ij, s_p1);
                m_ij = fmaxf(m_ij, s_p2);
            }

            // Update running max
            float m_new = fmaxf(m_i, m_ij);
            float alpha = expf(m_i - m_new);

            // Rescale previous accumulator
            l_i *= alpha;
            for (int d = 0; d < D; d++) {
                acc_smem[m * D + d] *= alpha;
            }

            // Accumulate for this K block
            // NOTE: Causal masking is already in the scores (invalid positions are NEG_INF).
            // However, we need to re-apply causal mask after convolution because
            // the convolution can bring in valid attention weights for V positions
            // that shouldn't receive any output.
            float l_ij = 0.0f;
            for (int n = 0; n < BLOCK_N; n++) {
                int v_pos = k_block_start + n;
                if (v_pos >= T_k) continue;

                // Get exp(QK - m_new) for all 5 shifts
                float s_m2 = S_smem[m * (BLOCK_N + 4) + n];
                float s_m1 = S_smem[m * (BLOCK_N + 4) + n + 1];
                float s_0  = S_smem[m * (BLOCK_N + 4) + n + 2];
                float s_p1 = S_smem[m * (BLOCK_N + 4) + n + 3];
                float s_p2 = S_smem[m * (BLOCK_N + 4) + n + 4];

                float p_m2 = (s_m2 > -1e20f) ? expf(s_m2 - m_new) : 0.0f;
                float p_m1 = (s_m1 > -1e20f) ? expf(s_m1 - m_new) : 0.0f;
                float p_0  = (s_0  > -1e20f) ? expf(s_0 - m_new) : 0.0f;
                float p_p1 = (s_p1 > -1e20f) ? expf(s_p1 - m_new) : 0.0f;
                float p_p2 = (s_p2 > -1e20f) ? expf(s_p2 - m_new) : 0.0f;

                // 5-tap convolution
                float p_conv = w0 * p_p2 + w1 * p_p1 + w2 * p_0 + w3 * p_m1 + w4 * p_m2;

                // Re-apply causal mask after convolution:
                // For causal attention, V at position v_pos should not receive
                // output if v_pos > q_pos. This is because even though some shifted
                // K positions might be valid, using V beyond the query position
                // would leak future information.
                if (IS_CAUSAL && v_pos > q_pos) {
                    p_conv = 0.0f;
                }

                l_ij += p_conv;

                // Accumulate output: acc += p_conv * V[n]
                for (int d = 0; d < D; d++) {
                    float v_val = bf16_to_float(V_smem[n * BLOCK_D + d]);
                    acc_smem[m * D + d] += p_conv * v_val;
                }
            }

            // Update running state
            l_smem[m] = l_i + l_ij;
            m_smem[m] = m_new;
        }
        __syncthreads();
    }

    // Write output - PARALLELIZED
    for (int m = tid; m < BLOCK_M; m += num_threads) {
        int q_pos = q_start + m;
        if (q_pos >= T_q) continue;

        float l_final = fmaxf(l_smem[m], 1e-9f);
        float m_final = m_smem[m];

        // Normalize and store output
        for (int d = 0; d < D; d++) {
            float o_val = acc_smem[m * D + d] / l_final;
            O_batch[q_pos * D + d] = float_to_bf16(o_val);
        }

        // Store LSE for backward pass
        LSE_batch[q_pos] = m_final + logf(l_final);
    }
}

// Wrapper to launch forward kernel
void launch_fused_look_around_flash_fwd(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const float* proj_weights,
    __nv_bfloat16* O,
    float* LSE,
    int B, int H, int T_q, int T_k, int D,
    float sm_scale,
    bool causal,
    cudaStream_t stream
) {
    // Choose block sizes based on head dimension
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;

    int num_q_blocks = (T_q + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_q_blocks, B * H);
    dim3 block(128);  // 4 warps

    // Calculate shared memory size
    // Q_smem: BLOCK_M * D * 2 bytes
    // K_smem: (BLOCK_N + 4) * D * 2 bytes
    // V_smem: BLOCK_N * D * 2 bytes
    // S_smem: BLOCK_M * (BLOCK_N + 4) * 4 bytes
    // m_smem: BLOCK_M * 4 bytes
    // l_smem: BLOCK_M * 4 bytes
    // acc_smem: BLOCK_M * D * 4 bytes
    size_t smem_size = 0;

    // Get device properties
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    if (D <= 64) {
        smem_size = BLOCK_M * 64 * 2 +           // Q_smem
                    (BLOCK_N + 4) * 64 * 2 +     // K_smem
                    BLOCK_N * 64 * 2 +           // V_smem
                    BLOCK_M * (BLOCK_N + 4) * 4 + // S_smem
                    BLOCK_M * 4 +                 // m_smem
                    BLOCK_M * 4 +                 // l_smem
                    BLOCK_M * 64 * 4 +           // acc_smem
                    5 * 4;                        // w_shared

        // Set max dynamic shared memory if needed
        if (smem_size > props.sharedMemPerBlock) {
            cudaFuncSetAttribute(
                causal ? fused_look_around_flash_fwd_kernel<BLOCK_M, BLOCK_N, 64, true>
                       : fused_look_around_flash_fwd_kernel<BLOCK_M, BLOCK_N, 64, false>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                smem_size
            );
        }

        if (causal) {
            fused_look_around_flash_fwd_kernel<BLOCK_M, BLOCK_N, 64, true>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, proj_weights, O, LSE,
                    B, H, T_q, T_k, D, sm_scale
                );
        } else {
            fused_look_around_flash_fwd_kernel<BLOCK_M, BLOCK_N, 64, false>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, proj_weights, O, LSE,
                    B, H, T_q, T_k, D, sm_scale
                );
        }
    } else {
        smem_size = BLOCK_M * 128 * 2 +          // Q_smem
                    (BLOCK_N + 4) * 128 * 2 +    // K_smem
                    BLOCK_N * 128 * 2 +          // V_smem
                    BLOCK_M * (BLOCK_N + 4) * 4 + // S_smem
                    BLOCK_M * 4 +                 // m_smem
                    BLOCK_M * 4 +                 // l_smem
                    BLOCK_M * 128 * 4 +          // acc_smem
                    5 * 4;                        // w_shared

        if (causal) {
            fused_look_around_flash_fwd_kernel<BLOCK_M, BLOCK_N, 128, true>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, proj_weights, O, LSE,
                    B, H, T_q, T_k, D, sm_scale
                );
        } else {
            fused_look_around_flash_fwd_kernel<BLOCK_M, BLOCK_N, 128, false>
                <<<grid, block, smem_size, stream>>>(
                    Q, K, V, proj_weights, O, LSE,
                    B, H, T_q, T_k, D, sm_scale
                );
        }
    }
}

}  // namespace fused_look_around
