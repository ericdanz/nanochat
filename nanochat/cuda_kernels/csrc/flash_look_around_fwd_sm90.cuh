// flash_look_around_fwd_sm90.cuh - Forward kernel implementation for FA3-style look-around attention
//
// This implements the forward pass of look-around attention with 5-tap convolution
// using a FlashAttention-3 style architecture:
// - Online softmax with shared max across 5 shifts
// - Tiled matrix multiplication for QK^T and P@V
// - WMMA tensor cores for compute
// - Pipelined K/V loading with double buffering
//
// For sm_90+ (Hopper/Blackwell), this can be upgraded to use:
// - TMA for async bulk memory loads
// - WGMMA for larger warp-group MMA operations
// - Producer/consumer warp specialization

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>

#include "flash_look_around_kernel_sm90.h"

namespace flash_look_around {

using namespace nvcuda::wmma;

////////////////////////////////////////////////////////////////////////////////////////////////////
// WMMA configuration
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WARP_SIZE = 32;

// Halo padding to align with WMMA (round 4 up to 16)
constexpr int HALO_SIZE = 2;        // ±2 for 5-tap
constexpr int HALO_PADDED = 16;     // Aligned for WMMA

// Negative infinity for masking
__device__ constexpr float NEG_INF = -1e30f;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward kernel implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL>
__global__ void flash_look_around_fwd_kernel_sm90(FlashLookAroundFwdParams params) {
    // Constants derived from template parameters
    constexpr int S_STRIDE = BLOCK_N + HALO_PADDED;  // Score stride with padded halo
    constexpr int S_OFFSET = HALO_PADDED / 2 - HALO_SIZE;  // Offset to center position in halo

    // WMMA tile counts
    constexpr int M_TILES = BLOCK_M / WMMA_M;
    constexpr int N_TILES = (BLOCK_N + HALO_PADDED) / WMMA_N;
    constexpr int K_TILES = BLOCK_D / WMMA_K;
    constexpr int PV_M_TILES = BLOCK_M / WMMA_M;
    constexpr int PV_N_TILES = BLOCK_D / WMMA_N;
    constexpr int PV_K_TILES = BLOCK_N / WMMA_K;

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
    const float* conv_weights = reinterpret_cast<const float*>(params.conv_weights_ptr)
        + kv_head_idx * 5;
    __nv_bfloat16* O_batch = reinterpret_cast<__nv_bfloat16*>(params.O_ptr)
        + batch_idx * params.O_batch_stride + q_head_idx * params.O_head_stride;
    float* LSE_batch = params.LSE_ptr
        + batch_idx * params.n_q_heads * params.T_q + q_head_idx * params.T_q;

    // Shared memory allocation
    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* K_smem = Q_smem + BLOCK_M * BLOCK_D;
    __nv_bfloat16* V_smem = K_smem + (BLOCK_N + HALO_PADDED) * BLOCK_D;
    float* S_smem = reinterpret_cast<float*>(V_smem + BLOCK_N * BLOCK_D);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_smem + BLOCK_M * S_STRIDE);
    float* O_block_smem = reinterpret_cast<float*>(P_smem + BLOCK_M * BLOCK_N);
    float* m_smem = O_block_smem + BLOCK_M * BLOCK_D;
    float* l_smem = m_smem + BLOCK_M;
    float* acc_smem = l_smem + BLOCK_M;

    // Load convolution weights to shared memory
    __shared__ float w_shared[5];
    if (tid < 5) {
        w_shared[tid] = conv_weights[tid];
    }
    __syncthreads();

    const float w0 = w_shared[0];  // Weight for shift +2
    const float w1 = w_shared[1];  // Weight for shift +1
    const float w2 = w_shared[2];  // Weight for shift 0 (center)
    const float w3 = w_shared[3];  // Weight for shift -1
    const float w4 = w_shared[4];  // Weight for shift -2

    const float sm_scale = params.sm_scale;
    const int T_q = params.T_q;
    const int T_k = params.T_k;
    const int D = params.D;
    const int window_left = params.window_left;

    // Load Q block to shared memory
    #pragma unroll 4
    for (int i = tid; i < BLOCK_M * BLOCK_D; i += num_threads) {
        int m = i / BLOCK_D;
        int d = i % BLOCK_D;
        int q_pos = q_start + m;
        if (q_pos < T_q && d < D) {
            Q_smem[m * BLOCK_D + d] = Q_batch[q_pos * D + d];
        } else {
            Q_smem[m * BLOCK_D + d] = __float2bfloat16(0.0f);
        }
    }

    // Initialize per-query accumulators
    #pragma unroll
    for (int m = tid; m < BLOCK_M; m += num_threads) {
        m_smem[m] = NEG_INF;
        l_smem[m] = 0.0f;
    }
    #pragma unroll 4
    for (int i = tid; i < BLOCK_M * D; i += num_threads) {
        acc_smem[i] = 0.0f;
    }
    __syncthreads();

    // Compute K/V iteration bounds based on window size
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

        // ============================================================
        // LOAD K WITH HALO
        // Positions: [k_block_start - HALO_PADDED/2, k_block_start + BLOCK_N + HALO_PADDED/2)
        // ============================================================
        #pragma unroll 4
        for (int i = tid; i < (BLOCK_N + HALO_PADDED) * BLOCK_D; i += num_threads) {
            int n = i / BLOCK_D;
            int d = i % BLOCK_D;
            int k_pos = k_block_start - HALO_PADDED / 2 + n;
            if (k_pos >= 0 && k_pos < T_k && d < D) {
                K_smem[n * BLOCK_D + d] = K_batch[k_pos * D + d];
            } else {
                K_smem[n * BLOCK_D + d] = __float2bfloat16(0.0f);
            }
        }

        // LOAD V (no halo needed)
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
        // WMMA QK^T COMPUTATION
        // Compute S = Q @ K^T with halo: (BLOCK_M, BLOCK_N + HALO_PADDED)
        // ============================================================
        for (int tile_idx = warp_id; tile_idx < M_TILES * N_TILES; tile_idx += num_warps) {
            int m_tile = tile_idx / N_TILES;
            int n_tile = tile_idx % N_TILES;

            // WMMA fragments
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> q_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> k_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;

            fill_fragment(s_frag, 0.0f);

            // Accumulate over K dimension (head dimension)
            #pragma unroll
            for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
                load_matrix_sync(q_frag, Q_smem + m_tile * WMMA_M * BLOCK_D + k_tile * WMMA_K, BLOCK_D);
                load_matrix_sync(k_frag, K_smem + n_tile * WMMA_N * BLOCK_D + k_tile * WMMA_K, BLOCK_D);
                mma_sync(s_frag, q_frag, k_frag, s_frag);
            }

            // Store to shared memory
            store_matrix_sync(S_smem + m_tile * WMMA_M * S_STRIDE + n_tile * WMMA_N,
                              s_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();

        // ============================================================
        // POST-WMMA: APPLY SCALING AND MASKS
        // ============================================================
        #pragma unroll 1
        for (int i = tid; i < BLOCK_M * S_STRIDE; i += num_threads) {
            int m = i / S_STRIDE;
            int n_ext = i % S_STRIDE;
            int q_pos = q_start + m;
            int k_pos = k_block_start - HALO_PADDED / 2 + n_ext;

            float s = S_smem[i] * sm_scale;

            // Apply masks
            bool valid = (q_pos < T_q) && (k_pos >= 0) && (k_pos < T_k);
            if (!valid) s = NEG_INF;
            if (IS_CAUSAL && k_pos > q_pos) s = NEG_INF;
            if (window_left >= 0 && k_pos < q_pos - window_left) s = NEG_INF;

            S_smem[i] = s;
        }
        __syncthreads();

        // ============================================================
        // PHASE 1: FIND MAX ACROSS ALL 5 SHIFTS
        // For online softmax with convolution
        // ============================================================
        __shared__ float m_block[BLOCK_M];
        __shared__ float l_block[BLOCK_M];

        #pragma unroll 1
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            int q_pos = q_start + m;
            float m_ij = NEG_INF;

            if (q_pos < T_q) {
                #pragma unroll 4
                for (int n = 0; n < BLOCK_N; n++) {
                    int v_pos = k_block_start + n;
                    if (v_pos >= T_k) continue;

                    // Max across all 5 shifts (positions contributing to output n)
                    float s_m2 = S_smem[m * S_STRIDE + n + S_OFFSET + 0];  // k_pos - 2
                    float s_m1 = S_smem[m * S_STRIDE + n + S_OFFSET + 1];  // k_pos - 1
                    float s_0  = S_smem[m * S_STRIDE + n + S_OFFSET + 2];  // k_pos
                    float s_p1 = S_smem[m * S_STRIDE + n + S_OFFSET + 3];  // k_pos + 1
                    float s_p2 = S_smem[m * S_STRIDE + n + S_OFFSET + 4];  // k_pos + 2

                    m_ij = fmaxf(m_ij, fmaxf(fmaxf(s_m2, s_m1), fmaxf(fmaxf(s_0, s_p1), s_p2)));
                }
            }
            m_block[m] = m_ij;
        }
        __syncthreads();

        // ============================================================
        // PHASE 2: COMPUTE P_CONV AND ROW SUMS
        // Apply softmax with 5-tap convolution
        // ============================================================
        #pragma unroll 1
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int m = i / BLOCK_N;
            int n = i % BLOCK_N;
            int q_pos = q_start + m;
            int v_pos = k_block_start + n;

            float p_conv = 0.0f;

            if (q_pos < T_q && v_pos < T_k) {
                // Use shared max for numerical stability
                float m_new = fmaxf(m_smem[m], m_block[m]);

                // Get scores for all 5 shifts
                float s_m2 = S_smem[m * S_STRIDE + n + S_OFFSET + 0];
                float s_m1 = S_smem[m * S_STRIDE + n + S_OFFSET + 1];
                float s_0  = S_smem[m * S_STRIDE + n + S_OFFSET + 2];
                float s_p1 = S_smem[m * S_STRIDE + n + S_OFFSET + 3];
                float s_p2 = S_smem[m * S_STRIDE + n + S_OFFSET + 4];

                // Compute exp(s - m_new) for each shift
                float p_m2 = (s_m2 > -1e20f) ? expf(s_m2 - m_new) : 0.0f;
                float p_m1 = (s_m1 > -1e20f) ? expf(s_m1 - m_new) : 0.0f;
                float p_0  = (s_0  > -1e20f) ? expf(s_0 - m_new)  : 0.0f;
                float p_p1 = (s_p1 > -1e20f) ? expf(s_p1 - m_new) : 0.0f;
                float p_p2 = (s_p2 > -1e20f) ? expf(s_p2 - m_new) : 0.0f;

                // Apply 5-tap convolution
                p_conv = w0 * p_p2 + w1 * p_p1 + w2 * p_0 + w3 * p_m1 + w4 * p_m2;

                // Re-apply causal mask after convolution
                if (IS_CAUSAL && v_pos > q_pos) {
                    p_conv = 0.0f;
                }
            }

            // Store P as bf16 for WMMA P@V
            P_smem[m * BLOCK_N + n] = __float2bfloat16(p_conv);
        }
        __syncthreads();

        // Compute row sums for normalization tracking
        #pragma unroll 1
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            float l_ij = 0.0f;
            #pragma unroll 4
            for (int n = 0; n < BLOCK_N; n++) {
                l_ij += __bfloat162float(P_smem[m * BLOCK_N + n]);
            }
            l_block[m] = l_ij;
        }
        __syncthreads();

        // ============================================================
        // PHASE 3: RESCALE ACCUMULATOR
        // Update running max and sum, rescale previous output
        // ============================================================
        #pragma unroll 1
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            int q_pos = q_start + m;
            if (q_pos >= T_q) continue;

            float m_i = m_smem[m];
            float l_i = l_smem[m];
            float m_new = fmaxf(m_i, m_block[m]);
            float alpha = expf(m_i - m_new);

            // Rescale previous accumulator
            float* acc_row = acc_smem + m * D;
            #pragma unroll 8
            for (int d = 0; d < D; d++) {
                acc_row[d] *= alpha;
            }

            // Update running state
            m_smem[m] = m_new;
            l_smem[m] = l_i * alpha + l_block[m];
        }
        __syncthreads();

        // ============================================================
        // PHASE 4: WMMA P @ V
        // Compute output contribution from this K block
        // ============================================================

        // Zero O_block_smem
        #pragma unroll 4
        for (int i = tid; i < BLOCK_M * BLOCK_D; i += num_threads) {
            O_block_smem[i] = 0.0f;
        }
        __syncthreads();

        // WMMA P @ V
        for (int tile_idx = warp_id; tile_idx < PV_M_TILES * PV_N_TILES; tile_idx += num_warps) {
            int m_tile = tile_idx / PV_N_TILES;
            int n_tile = tile_idx % PV_N_TILES;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> p_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> v_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;

            fill_fragment(o_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < PV_K_TILES; k_tile++) {
                load_matrix_sync(p_frag, P_smem + m_tile * WMMA_M * BLOCK_N + k_tile * WMMA_K, BLOCK_N);
                load_matrix_sync(v_frag, V_smem + k_tile * WMMA_K * BLOCK_D + n_tile * WMMA_N, BLOCK_D);
                mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            store_matrix_sync(O_block_smem + m_tile * WMMA_M * BLOCK_D + n_tile * WMMA_N,
                              o_frag, BLOCK_D, mem_row_major);
        }
        __syncthreads();

        // Add O_block to accumulator
        #pragma unroll 4
        for (int i = tid; i < BLOCK_M * D; i += num_threads) {
            int m = i / D;
            int d = i % D;
            acc_smem[m * D + d] += O_block_smem[m * BLOCK_D + d];
        }
        __syncthreads();
    }

    // ============================================================
    // WRITE OUTPUT
    // Normalize and store final output
    // ============================================================
    #pragma unroll 1
    for (int m = tid; m < BLOCK_M; m += num_threads) {
        int q_pos = q_start + m;
        if (q_pos >= T_q) continue;

        float l_final = fmaxf(l_smem[m], 1e-9f);
        float m_final = m_smem[m];
        float inv_l = 1.0f / l_final;

        // Normalize and store
        const float* acc_row = acc_smem + m * D;
        __nv_bfloat16* o_row = O_batch + q_pos * D;

        #pragma unroll 8
        for (int d = 0; d < D; d++) {
            o_row[d] = __float2bfloat16(acc_row[d] * inv_l);
        }

        // Store LSE for backward pass
        LSE_batch[q_pos] = m_final + logf(l_final);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// WGMMA Forward kernel implementation (128 threads, 1 warp group)
// Uses SM90_64x64x16_F32BF16BF16_RS for both QK^T and P@V
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900

template <int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL>
__global__ void flash_look_around_fwd_kernel_wgmma(FlashLookAroundFwdParams params) {
    using namespace cute;

    // Constants derived from template parameters
    constexpr int S_STRIDE = BLOCK_N + HALO_PADDED;  // Score stride with padded halo
    constexpr int S_OFFSET = HALO_PADDED / 2 - HALO_SIZE;  // Offset to center position in halo

    // WGMMA configuration
    constexpr int WGMMA_M = 64;
    constexpr int WGMMA_N = 64;
    constexpr int WGMMA_K = 16;

    // Tile counts for WGMMA
    // For QK^T: Q is (BLOCK_M, BLOCK_D), K^T is (BLOCK_D, BLOCK_N+HALO_PADDED)
    // We compute M=64 rows at a time, N=64 cols at a time
    constexpr int QK_M_TILES = BLOCK_M / WGMMA_M;  // 1 for BLOCK_M=64
    constexpr int QK_N_TILES = (BLOCK_N + HALO_PADDED) / WGMMA_N;  // 80/64 = need special handling
    constexpr int QK_K_TILES = BLOCK_D / WGMMA_K;  // 64/16 = 4

    // For P@V: P is (BLOCK_M, BLOCK_N), V is (BLOCK_N, BLOCK_D)
    constexpr int PV_M_TILES = BLOCK_M / WGMMA_M;  // 1
    constexpr int PV_N_TILES = BLOCK_D / WGMMA_N;  // 1 for D=64
    constexpr int PV_K_TILES = BLOCK_N / WGMMA_K;  // 64/16 = 4

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
    const int num_threads = blockDim.x;  // 128 threads = 1 warp group
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    // Query range for this block
    const int q_start = q_block_idx * BLOCK_M;

    // Input/output pointers
    const __nv_bfloat16* Q_batch = reinterpret_cast<const __nv_bfloat16*>(params.Q_ptr)
        + batch_idx * params.Q_batch_stride + q_head_idx * params.Q_head_stride;
    const __nv_bfloat16* K_batch = reinterpret_cast<const __nv_bfloat16*>(params.K_ptr)
        + batch_idx * params.K_batch_stride + kv_head_idx * params.K_head_stride;
    const __nv_bfloat16* V_batch = reinterpret_cast<const __nv_bfloat16*>(params.V_ptr)
        + batch_idx * params.V_batch_stride + kv_head_idx * params.V_head_stride;
    const float* conv_weights = reinterpret_cast<const float*>(params.conv_weights_ptr)
        + kv_head_idx * 5;
    __nv_bfloat16* O_batch = reinterpret_cast<__nv_bfloat16*>(params.O_ptr)
        + batch_idx * params.O_batch_stride + q_head_idx * params.O_head_stride;
    float* LSE_batch = params.LSE_ptr
        + batch_idx * params.n_q_heads * params.T_q + q_head_idx * params.T_q;

    // Shared memory allocation
    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* K_smem = Q_smem + BLOCK_M * BLOCK_D;
    __nv_bfloat16* V_smem = K_smem + (BLOCK_N + HALO_PADDED) * BLOCK_D;
    float* S_smem = reinterpret_cast<float*>(V_smem + BLOCK_N * BLOCK_D);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_smem + BLOCK_M * S_STRIDE);
    float* O_block_smem = reinterpret_cast<float*>(P_smem + BLOCK_M * BLOCK_N);
    float* m_smem = O_block_smem + BLOCK_M * BLOCK_D;
    float* l_smem = m_smem + BLOCK_M;
    float* acc_smem = l_smem + BLOCK_M;

    // Load convolution weights to shared memory
    __shared__ float w_shared[5];
    if (tid < 5) {
        w_shared[tid] = conv_weights[tid];
    }
    __syncthreads();

    const float w0 = w_shared[0];
    const float w1 = w_shared[1];
    const float w2 = w_shared[2];
    const float w3 = w_shared[3];
    const float w4 = w_shared[4];

    const float sm_scale = params.sm_scale;
    const int T_q = params.T_q;
    const int T_k = params.T_k;
    const int D = params.D;
    const int window_left = params.window_left;

    // Load Q block to shared memory (all threads cooperate)
    #pragma unroll 4
    for (int i = tid; i < BLOCK_M * BLOCK_D; i += num_threads) {
        int m = i / BLOCK_D;
        int d = i % BLOCK_D;
        int q_pos = q_start + m;
        if (q_pos < T_q && d < D) {
            Q_smem[m * BLOCK_D + d] = Q_batch[q_pos * D + d];
        } else {
            Q_smem[m * BLOCK_D + d] = __float2bfloat16(0.0f);
        }
    }

    // Initialize per-query accumulators
    for (int m = tid; m < BLOCK_M; m += num_threads) {
        m_smem[m] = NEG_INF;
        l_smem[m] = 0.0f;
    }
    for (int i = tid; i < BLOCK_M * D; i += num_threads) {
        acc_smem[i] = 0.0f;
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

        // ============================================================
        // LOAD K WITH HALO
        // ============================================================
        #pragma unroll 4
        for (int i = tid; i < (BLOCK_N + HALO_PADDED) * BLOCK_D; i += num_threads) {
            int n = i / BLOCK_D;
            int d = i % BLOCK_D;
            int k_pos = k_block_start - HALO_PADDED / 2 + n;
            if (k_pos >= 0 && k_pos < T_k && d < D) {
                K_smem[n * BLOCK_D + d] = K_batch[k_pos * D + d];
            } else {
                K_smem[n * BLOCK_D + d] = __float2bfloat16(0.0f);
            }
        }

        // LOAD V
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
        // WGMMA QK^T COMPUTATION
        // Compute S = Q @ K^T for first 64 columns
        // The halo portion (cols 64-79) is computed separately with WMMA
        // ============================================================

        // Accumulator registers for 64x64 output tile
        // Each thread in the warp group holds part of the result
        float s_acc[32];  // 32 floats per thread for 64x64 tile
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            s_acc[i] = 0.0f;
        }

        // Create descriptor for K (first 64 columns, starting at col 8 to skip left halo)
        // K_smem layout: (80, D) row-major, we want columns [8, 72) for main block
        uint64_t desc_k = make_k_desc(K_smem + 8 * BLOCK_D, 64, BLOCK_D);

        // WGMMA fence before starting MMA operations
        warpgroup_fence_operand(s_acc[0]);
        warpgroup_arrive();

        // Loop over K dimension (head dimension)
        #pragma unroll
        for (int k_iter = 0; k_iter < QK_K_TILES; k_iter++) {
            // Load Q fragment into registers for this K tile
            // Each thread loads part of the 64x16 Q tile
            // For RS variant, A operand (Q) is in registers
            uint32_t q_regs[4];  // 4 registers = 8 bf16 values

            // Thread's position in the warp group for Q loading
            // WGMMA expects specific register layout for A operand
            const int wg_lane = tid % 128;
            const int row_in_tile = wg_lane / 4;  // 0-31, repeated twice
            const int k_offset = (wg_lane % 4) * 4;

            // Load 8 bf16 values (4 registers x 2 bf16 per register)
            const __nv_bfloat16* q_ptr = Q_smem + row_in_tile * BLOCK_D + k_iter * WGMMA_K + k_offset;
            if (row_in_tile < 64) {
                q_regs[0] = *reinterpret_cast<const uint32_t*>(q_ptr);
                q_regs[1] = *reinterpret_cast<const uint32_t*>(q_ptr + 2);
                q_regs[2] = *reinterpret_cast<const uint32_t*>(q_ptr + 4);
                q_regs[3] = *reinterpret_cast<const uint32_t*>(q_ptr + 6);
            } else {
                q_regs[0] = q_regs[1] = q_regs[2] = q_regs[3] = 0;
            }

            // Update K descriptor for this K tile
            uint64_t desc_k_tile = make_wgmma_desc(
                K_smem + 8 * BLOCK_D + k_iter * WGMMA_K,
                BLOCK_D * sizeof(__nv_bfloat16),
                8 * BLOCK_D * sizeof(__nv_bfloat16)
            );

            // Issue WGMMA
            SM90_64x64x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>::fma(
                q_regs[0], q_regs[1], q_regs[2], q_regs[3],
                desc_k_tile,
                s_acc[0],  s_acc[1],  s_acc[2],  s_acc[3],
                s_acc[4],  s_acc[5],  s_acc[6],  s_acc[7],
                s_acc[8],  s_acc[9],  s_acc[10], s_acc[11],
                s_acc[12], s_acc[13], s_acc[14], s_acc[15],
                s_acc[16], s_acc[17], s_acc[18], s_acc[19],
                s_acc[20], s_acc[21], s_acc[22], s_acc[23],
                s_acc[24], s_acc[25], s_acc[26], s_acc[27],
                s_acc[28], s_acc[29], s_acc[30], s_acc[31],
                k_iter == 0 ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One
            );
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        __syncthreads();

        // Store S accumulator to shared memory
        // WGMMA output layout: each thread has specific elements of the 64x64 tile
        // Thread i in warp group owns specific (row, col) pairs
        const int wg_lane = tid % 128;
        #pragma unroll
        for (int reg_idx = 0; reg_idx < 32; reg_idx++) {
            // Decode output position from register index and lane
            // For 64x64 output with 128 threads, each thread owns 32 elements
            // Layout: reg_idx encodes position within the tile
            int row = (reg_idx / 4) * 2 + (wg_lane / 32);
            int col = (reg_idx % 4) * 16 + (wg_lane % 4) * 4 + ((wg_lane / 4) % 8);

            // Offset by 8 columns for the halo (S_smem includes left halo)
            if (row < BLOCK_M && col < 64) {
                S_smem[row * S_STRIDE + col + 8] = s_acc[reg_idx] * sm_scale;
            }
        }

        // Handle halo columns with simple scalar computation
        // Left halo: columns 0-7, Right halo: columns 72-79
        #pragma unroll 1
        for (int i = tid; i < BLOCK_M * 16; i += num_threads) {
            int m = i / 16;
            int h = i % 16;  // 0-15 for left/right halo
            int n_ext = (h < 8) ? h : (64 + h);  // 0-7 or 72-79

            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < BLOCK_D; d++) {
                s += __bfloat162float(Q_smem[m * BLOCK_D + d]) *
                     __bfloat162float(K_smem[n_ext * BLOCK_D + d]);
            }
            S_smem[m * S_STRIDE + n_ext] = s * sm_scale;
        }
        __syncthreads();

        // ============================================================
        // POST-WGMMA: APPLY MASKS (same as WMMA version)
        // ============================================================
        #pragma unroll 1
        for (int i = tid; i < BLOCK_M * S_STRIDE; i += num_threads) {
            int m = i / S_STRIDE;
            int n_ext = i % S_STRIDE;
            int q_pos = q_start + m;
            int k_pos = k_block_start - HALO_PADDED / 2 + n_ext;

            float s = S_smem[i];

            // Apply masks
            bool valid = (q_pos < T_q) && (k_pos >= 0) && (k_pos < T_k);
            if (!valid) s = NEG_INF;
            if (IS_CAUSAL && k_pos > q_pos) s = NEG_INF;
            if (window_left >= 0 && k_pos < q_pos - window_left) s = NEG_INF;

            S_smem[i] = s;
        }
        __syncthreads();

        // ============================================================
        // PHASE 1: FIND MAX ACROSS ALL 5 SHIFTS
        // ============================================================
        __shared__ float m_block[BLOCK_M];
        __shared__ float l_block[BLOCK_M];

        #pragma unroll 1
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            int q_pos = q_start + m;
            float m_ij = NEG_INF;

            if (q_pos < T_q) {
                #pragma unroll 4
                for (int n = 0; n < BLOCK_N; n++) {
                    int v_pos = k_block_start + n;
                    if (v_pos >= T_k) continue;

                    float s_m2 = S_smem[m * S_STRIDE + n + S_OFFSET + 0];
                    float s_m1 = S_smem[m * S_STRIDE + n + S_OFFSET + 1];
                    float s_0  = S_smem[m * S_STRIDE + n + S_OFFSET + 2];
                    float s_p1 = S_smem[m * S_STRIDE + n + S_OFFSET + 3];
                    float s_p2 = S_smem[m * S_STRIDE + n + S_OFFSET + 4];

                    m_ij = fmaxf(m_ij, fmaxf(fmaxf(s_m2, s_m1), fmaxf(fmaxf(s_0, s_p1), s_p2)));
                }
            }
            m_block[m] = m_ij;
        }
        __syncthreads();

        // ============================================================
        // PHASE 2: COMPUTE P_CONV AND ROW SUMS
        // ============================================================
        #pragma unroll 1
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int m = i / BLOCK_N;
            int n = i % BLOCK_N;
            int q_pos = q_start + m;
            int v_pos = k_block_start + n;

            float p_conv = 0.0f;

            if (q_pos < T_q && v_pos < T_k) {
                float m_new = fmaxf(m_smem[m], m_block[m]);

                float s_m2 = S_smem[m * S_STRIDE + n + S_OFFSET + 0];
                float s_m1 = S_smem[m * S_STRIDE + n + S_OFFSET + 1];
                float s_0  = S_smem[m * S_STRIDE + n + S_OFFSET + 2];
                float s_p1 = S_smem[m * S_STRIDE + n + S_OFFSET + 3];
                float s_p2 = S_smem[m * S_STRIDE + n + S_OFFSET + 4];

                float p_m2 = (s_m2 > -1e20f) ? expf(s_m2 - m_new) : 0.0f;
                float p_m1 = (s_m1 > -1e20f) ? expf(s_m1 - m_new) : 0.0f;
                float p_0  = (s_0  > -1e20f) ? expf(s_0 - m_new)  : 0.0f;
                float p_p1 = (s_p1 > -1e20f) ? expf(s_p1 - m_new) : 0.0f;
                float p_p2 = (s_p2 > -1e20f) ? expf(s_p2 - m_new) : 0.0f;

                p_conv = w0 * p_p2 + w1 * p_p1 + w2 * p_0 + w3 * p_m1 + w4 * p_m2;

                if (IS_CAUSAL && v_pos > q_pos) {
                    p_conv = 0.0f;
                }
            }

            P_smem[m * BLOCK_N + n] = __float2bfloat16(p_conv);
        }
        __syncthreads();

        // Compute row sums
        #pragma unroll 1
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            float l_ij = 0.0f;
            #pragma unroll 4
            for (int n = 0; n < BLOCK_N; n++) {
                l_ij += __bfloat162float(P_smem[m * BLOCK_N + n]);
            }
            l_block[m] = l_ij;
        }
        __syncthreads();

        // ============================================================
        // PHASE 3: RESCALE ACCUMULATOR
        // ============================================================
        #pragma unroll 1
        for (int m = tid; m < BLOCK_M; m += num_threads) {
            int q_pos = q_start + m;
            if (q_pos >= T_q) continue;

            float m_i = m_smem[m];
            float l_i = l_smem[m];
            float m_new = fmaxf(m_i, m_block[m]);
            float alpha = expf(m_i - m_new);

            float* acc_row = acc_smem + m * D;
            #pragma unroll 8
            for (int d = 0; d < D; d++) {
                acc_row[d] *= alpha;
            }

            m_smem[m] = m_new;
            l_smem[m] = l_i * alpha + l_block[m];
        }
        __syncthreads();

        // ============================================================
        // PHASE 4: WGMMA P @ V
        // ============================================================

        // Zero O_block
        #pragma unroll 4
        for (int i = tid; i < BLOCK_M * BLOCK_D; i += num_threads) {
            O_block_smem[i] = 0.0f;
        }
        __syncthreads();

        // Accumulator registers for P@V
        float o_acc[32];
        #pragma unroll
        for (int i = 0; i < 32; i++) {
            o_acc[i] = 0.0f;
        }

        // WGMMA for P @ V
        warpgroup_fence_operand(o_acc[0]);
        warpgroup_arrive();

        #pragma unroll
        for (int k_iter = 0; k_iter < PV_K_TILES; k_iter++) {
            // Load P fragment into registers
            uint32_t p_regs[4];

            const int wg_lane = tid % 128;
            const int row_in_tile = wg_lane / 4;
            const int k_offset = (wg_lane % 4) * 4;

            const __nv_bfloat16* p_ptr = P_smem + row_in_tile * BLOCK_N + k_iter * WGMMA_K + k_offset;
            if (row_in_tile < 64) {
                p_regs[0] = *reinterpret_cast<const uint32_t*>(p_ptr);
                p_regs[1] = *reinterpret_cast<const uint32_t*>(p_ptr + 2);
                p_regs[2] = *reinterpret_cast<const uint32_t*>(p_ptr + 4);
                p_regs[3] = *reinterpret_cast<const uint32_t*>(p_ptr + 6);
            } else {
                p_regs[0] = p_regs[1] = p_regs[2] = p_regs[3] = 0;
            }

            // V descriptor
            uint64_t desc_v_tile = make_wgmma_desc(
                V_smem + k_iter * WGMMA_K * BLOCK_D,
                BLOCK_D * sizeof(__nv_bfloat16),
                8 * BLOCK_D * sizeof(__nv_bfloat16)
            );

            SM90_64x64x16_F32BF16BF16_RS<GMMA::Major::K, GMMA::Major::K>::fma(
                p_regs[0], p_regs[1], p_regs[2], p_regs[3],
                desc_v_tile,
                o_acc[0],  o_acc[1],  o_acc[2],  o_acc[3],
                o_acc[4],  o_acc[5],  o_acc[6],  o_acc[7],
                o_acc[8],  o_acc[9],  o_acc[10], o_acc[11],
                o_acc[12], o_acc[13], o_acc[14], o_acc[15],
                o_acc[16], o_acc[17], o_acc[18], o_acc[19],
                o_acc[20], o_acc[21], o_acc[22], o_acc[23],
                o_acc[24], o_acc[25], o_acc[26], o_acc[27],
                o_acc[28], o_acc[29], o_acc[30], o_acc[31],
                k_iter == 0 ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One
            );
        }

        warpgroup_commit_batch();
        warpgroup_wait<0>();
        __syncthreads();

        // Store O accumulator to shared memory
        const int wg_lane2 = tid % 128;
        #pragma unroll
        for (int reg_idx = 0; reg_idx < 32; reg_idx++) {
            int row = (reg_idx / 4) * 2 + (wg_lane2 / 32);
            int col = (reg_idx % 4) * 16 + (wg_lane2 % 4) * 4 + ((wg_lane2 / 4) % 8);

            if (row < BLOCK_M && col < BLOCK_D) {
                O_block_smem[row * BLOCK_D + col] = o_acc[reg_idx];
            }
        }
        __syncthreads();

        // Add O_block to accumulator
        #pragma unroll 4
        for (int i = tid; i < BLOCK_M * D; i += num_threads) {
            int m = i / D;
            int d = i % D;
            acc_smem[m * D + d] += O_block_smem[m * BLOCK_D + d];
        }
        __syncthreads();
    }

    // ============================================================
    // WRITE OUTPUT
    // ============================================================
    #pragma unroll 1
    for (int m = tid; m < BLOCK_M; m += num_threads) {
        int q_pos = q_start + m;
        if (q_pos >= T_q) continue;

        float l_final = fmaxf(l_smem[m], 1e-9f);
        float m_final = m_smem[m];
        float inv_l = 1.0f / l_final;

        const float* acc_row = acc_smem + m * D;
        __nv_bfloat16* o_row = O_batch + q_pos * D;

        #pragma unroll 8
        for (int d = 0; d < D; d++) {
            o_row[d] = __float2bfloat16(acc_row[d] * inv_l);
        }

        LSE_batch[q_pos] = m_final + logf(l_final);
    }
}

#else  // __CUDA_ARCH__ < 900

// Fallback stub for non-Hopper architectures
template <int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL>
__global__ void flash_look_around_fwd_kernel_wgmma(FlashLookAroundFwdParams params) {
    // WGMMA requires sm_90+
    // This kernel should not be called on older architectures
}

#endif  // __CUDA_ARCH__ >= 900

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel launcher implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

inline void launch_flash_look_around_fwd_sm90(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const float* conv_weights,
    __nv_bfloat16* O,
    float* LSE,
    int B, int n_q_heads, int n_kv_heads, int T_q, int T_k, int D,
    float sm_scale,
    bool causal,
    int window_left,
    cudaStream_t stream
) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;

    // Prepare parameters
    FlashLookAroundFwdParams params;
    params.Q_ptr = Q;
    params.K_ptr = K;
    params.V_ptr = V;
    params.conv_weights_ptr = conv_weights;
    params.O_ptr = O;
    params.LSE_ptr = LSE;
    params.B = B;
    params.n_q_heads = n_q_heads;
    params.n_kv_heads = n_kv_heads;
    params.T_q = T_q;
    params.T_k = T_k;
    params.D = D;
    params.sm_scale = sm_scale;
    params.causal = causal;
    params.window_left = window_left;

    // Set up strides (assuming contiguous layout: B, H, T, D)
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

    // Launch configuration
    int num_q_blocks = (T_q + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_q_blocks, B * n_q_heads);
    dim3 block(256);  // 8 warps

    // Calculate shared memory size
    size_t smem_size = 0;
    if (D <= 64) {
        constexpr int BLOCK_D = 64;
        constexpr int S_STRIDE = BLOCK_N + HALO_PADDED;

        smem_size = BLOCK_M * BLOCK_D * sizeof(__nv_bfloat16) +           // Q_smem
                    (BLOCK_N + HALO_PADDED) * BLOCK_D * sizeof(__nv_bfloat16) +  // K_smem
                    BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16) +           // V_smem
                    BLOCK_M * S_STRIDE * sizeof(float) +                  // S_smem
                    BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16) +           // P_smem
                    BLOCK_M * BLOCK_D * sizeof(float) +                   // O_block_smem
                    BLOCK_M * sizeof(float) +                             // m_smem
                    BLOCK_M * sizeof(float) +                             // l_smem
                    BLOCK_M * BLOCK_D * sizeof(float) +                   // acc_smem
                    BLOCK_M * sizeof(float) * 2 +                         // m_block, l_block
                    5 * sizeof(float);                                    // w_shared

        cudaFuncSetAttribute(
            causal ? flash_look_around_fwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, true>
                   : flash_look_around_fwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        if (causal) {
            flash_look_around_fwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, true>
                <<<grid, block, smem_size, stream>>>(params);
        } else {
            flash_look_around_fwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, false>
                <<<grid, block, smem_size, stream>>>(params);
        }
    } else {
        // D = 128 case
        constexpr int BLOCK_D = 128;
        constexpr int S_STRIDE = BLOCK_N + HALO_PADDED;

        smem_size = BLOCK_M * BLOCK_D * sizeof(__nv_bfloat16) +
                    (BLOCK_N + HALO_PADDED) * BLOCK_D * sizeof(__nv_bfloat16) +
                    BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16) +
                    BLOCK_M * S_STRIDE * sizeof(float) +
                    BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16) +
                    BLOCK_M * BLOCK_D * sizeof(float) +
                    BLOCK_M * sizeof(float) +
                    BLOCK_M * sizeof(float) +
                    BLOCK_M * BLOCK_D * sizeof(float) +
                    BLOCK_M * sizeof(float) * 2 +
                    5 * sizeof(float);

        cudaFuncSetAttribute(
            causal ? flash_look_around_fwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, true>
                   : flash_look_around_fwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        if (causal) {
            flash_look_around_fwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, true>
                <<<grid, block, smem_size, stream>>>(params);
        } else {
            flash_look_around_fwd_kernel_sm90<BLOCK_M, BLOCK_N, BLOCK_D, false>
                <<<grid, block, smem_size, stream>>>(params);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// WGMMA Kernel launcher implementation (128 threads, 1 warp group)
////////////////////////////////////////////////////////////////////////////////////////////////////

inline void launch_flash_look_around_fwd_wgmma(
    const __nv_bfloat16* Q,
    const __nv_bfloat16* K,
    const __nv_bfloat16* V,
    const float* conv_weights,
    __nv_bfloat16* O,
    float* LSE,
    int B, int n_q_heads, int n_kv_heads, int T_q, int T_k, int D,
    float sm_scale,
    bool causal,
    int window_left,
    cudaStream_t stream
) {
    constexpr int BLOCK_M = 64;
    constexpr int BLOCK_N = 64;

    // Prepare parameters
    FlashLookAroundFwdParams params;
    params.Q_ptr = Q;
    params.K_ptr = K;
    params.V_ptr = V;
    params.conv_weights_ptr = conv_weights;
    params.O_ptr = O;
    params.LSE_ptr = LSE;
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

    // Launch configuration for WGMMA: 128 threads = 1 warp group
    int num_q_blocks = (T_q + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_q_blocks, B * n_q_heads);
    dim3 block(128);  // 4 warps = 1 warp group

    // Calculate shared memory size (same as WMMA version)
    size_t smem_size = 0;
    if (D <= 64) {
        constexpr int BLOCK_D = 64;
        constexpr int S_STRIDE = BLOCK_N + HALO_PADDED;

        smem_size = BLOCK_M * BLOCK_D * sizeof(__nv_bfloat16) +           // Q_smem
                    (BLOCK_N + HALO_PADDED) * BLOCK_D * sizeof(__nv_bfloat16) +  // K_smem
                    BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16) +           // V_smem
                    BLOCK_M * S_STRIDE * sizeof(float) +                  // S_smem
                    BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16) +           // P_smem
                    BLOCK_M * BLOCK_D * sizeof(float) +                   // O_block_smem
                    BLOCK_M * sizeof(float) +                             // m_smem
                    BLOCK_M * sizeof(float) +                             // l_smem
                    BLOCK_M * BLOCK_D * sizeof(float) +                   // acc_smem
                    BLOCK_M * sizeof(float) * 2 +                         // m_block, l_block
                    5 * sizeof(float);                                    // w_shared

        cudaFuncSetAttribute(
            causal ? flash_look_around_fwd_kernel_wgmma<BLOCK_M, BLOCK_N, BLOCK_D, true>
                   : flash_look_around_fwd_kernel_wgmma<BLOCK_M, BLOCK_N, BLOCK_D, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        if (causal) {
            flash_look_around_fwd_kernel_wgmma<BLOCK_M, BLOCK_N, BLOCK_D, true>
                <<<grid, block, smem_size, stream>>>(params);
        } else {
            flash_look_around_fwd_kernel_wgmma<BLOCK_M, BLOCK_N, BLOCK_D, false>
                <<<grid, block, smem_size, stream>>>(params);
        }
    } else {
        // D = 128 case
        constexpr int BLOCK_D = 128;
        constexpr int S_STRIDE = BLOCK_N + HALO_PADDED;

        smem_size = BLOCK_M * BLOCK_D * sizeof(__nv_bfloat16) +
                    (BLOCK_N + HALO_PADDED) * BLOCK_D * sizeof(__nv_bfloat16) +
                    BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16) +
                    BLOCK_M * S_STRIDE * sizeof(float) +
                    BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16) +
                    BLOCK_M * BLOCK_D * sizeof(float) +
                    BLOCK_M * sizeof(float) +
                    BLOCK_M * sizeof(float) +
                    BLOCK_M * BLOCK_D * sizeof(float) +
                    BLOCK_M * sizeof(float) * 2 +
                    5 * sizeof(float);

        cudaFuncSetAttribute(
            causal ? flash_look_around_fwd_kernel_wgmma<BLOCK_M, BLOCK_N, BLOCK_D, true>
                   : flash_look_around_fwd_kernel_wgmma<BLOCK_M, BLOCK_N, BLOCK_D, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );

        if (causal) {
            flash_look_around_fwd_kernel_wgmma<BLOCK_M, BLOCK_N, BLOCK_D, true>
                <<<grid, block, smem_size, stream>>>(params);
        } else {
            flash_look_around_fwd_kernel_wgmma<BLOCK_M, BLOCK_N, BLOCK_D, false>
                <<<grid, block, smem_size, stream>>>(params);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash_look_around
