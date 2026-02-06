// flash_look_around_fwd_sm120.cuh - Forward kernel implementation for SM120 (Blackwell)
//
// This implements the forward pass of look-around attention with 5-tap convolution
// using Blackwell's tcgen05 MMA instructions and TMEM:
// - Online softmax with shared max across 5 shifts
// - Tiled matrix multiplication for QK^T and P@V via tcgen05.mma
// - Accumulator in TMEM (Tensor Memory) instead of registers
// - 3-stage pipelining for better memory hiding
//
// Key differences from SM90 WGMMA:
// - Single-thread MMA issue model (elect_one_sync)
// - TMEM accumulator storage (256KB per SM)
// - Uses SM100_MMA_F16BF16_SS for both operands in SMEM

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>

#include "flash_look_around_kernel_sm120.h"

namespace flash_look_around {

using namespace nvcuda::wmma;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Configuration
////////////////////////////////////////////////////////////////////////////////////////////////////

constexpr int SM120_WARP_SIZE = 32;
constexpr int SM120_HALO_SIZE = 2;        // ±2 for 5-tap
constexpr int SM120_HALO_PADDED = 16;     // Aligned for MMA

// Negative infinity for masking
__device__ constexpr float SM120_NEG_INF = -1e30f;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Simple test kernel to verify launch mechanism (NON-TEMPLATE version)
////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void test_sm120_simple_kernel(__nv_bfloat16* O) {
    // Just write a simple value to output to verify launch works
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        O[0] = __float2bfloat16(42.0f);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Simple test kernel to verify launch mechanism
////////////////////////////////////////////////////////////////////////////////////////////////////
template <int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL>
__global__ void test_sm120_kernel(FlashLookAroundFwdParams params) {
    // Just write a simple value to output to verify launch works
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        __nv_bfloat16* O = reinterpret_cast<__nv_bfloat16*>(params.O_ptr);
        O[0] = __float2bfloat16(42.0f);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SM120 Forward Kernel
// Uses WMMA as fallback for initial implementation, with tcgen05 MMA path for CUDA 13.0+
// Note: This kernel is only compiled when ENABLE_SM120 is set, so we don't need CUDA_ARCH guards
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL>
__global__ void flash_look_around_fwd_kernel_sm120(FlashLookAroundFwdParams params) {
    // Constants
    constexpr int S_STRIDE = BLOCK_N + SM120_HALO_PADDED;
    constexpr int S_OFFSET = SM120_HALO_PADDED / 2 - SM120_HALO_SIZE;

    // WMMA tile configuration (fallback path using WMMA for initial implementation)
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    constexpr int M_TILES = BLOCK_M / WMMA_M;
    constexpr int N_TILES = (BLOCK_N + SM120_HALO_PADDED) / WMMA_N;
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
    const int warp_id = tid / SM120_WARP_SIZE;
    const int lane_id = tid % SM120_WARP_SIZE;
    const int num_warps = num_threads / SM120_WARP_SIZE;

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

    // Shared memory allocation (single-buffered K/V for RTX 5090's 99KB limit)
    // Layout (O_block removed to fit F8 buffers within 99KB):
    // - Q: BLOCK_M * BLOCK_D (loaded once)
    // - K: (BLOCK_N + SM120_HALO_PADDED) * BLOCK_D (single buffer with halo)
    // - V: BLOCK_N * BLOCK_D (single buffer)
    // - S, P, m, l, acc: single buffers (recomputed each iteration)
    // - F8 buffers (Q_e4m3, K_e4m3, P_e4m3, V_e4m3): for F8 MMA path
    // Total: ~88KB with F8 buffers (fits in 99KB opt-in limit)
    constexpr int K_STAGE_ELEMS = (BLOCK_N + SM120_HALO_PADDED) * BLOCK_D;
    constexpr int V_STAGE_ELEMS = BLOCK_N * BLOCK_D;

    extern __shared__ char smem[];
    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    __nv_bfloat16* K_smem = Q_smem + BLOCK_M * BLOCK_D;                    // Single-buffered K
    __nv_bfloat16* V_smem = K_smem + K_STAGE_ELEMS;                        // Single-buffered V
    float* S_smem = reinterpret_cast<float*>(V_smem + V_STAGE_ELEMS);
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_smem + BLOCK_M * S_STRIDE);
    // O_block_smem removed - accumulate directly to acc_smem
    float* m_smem = reinterpret_cast<float*>(P_smem + BLOCK_M * BLOCK_N);
    float* l_smem = m_smem + BLOCK_M;
    float* acc_smem = l_smem + BLOCK_M;

#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
    // F8 MMA shared memory for quantized Q, K, P, V
    // Note: K_e4m3 and V_e4m3 are single buffers for the current stage's quantized data
    // Aligned to 128 bytes for optimal access
    uint8_t* Q_e4m3 = reinterpret_cast<uint8_t*>(acc_smem + BLOCK_M * BLOCK_D);
    uint8_t* K_e4m3 = Q_e4m3 + BLOCK_M * BLOCK_D;
    uint8_t* P_e4m3 = K_e4m3 + (BLOCK_N + SM120_HALO_PADDED) * BLOCK_D;
    uint8_t* V_e4m3 = P_e4m3 + BLOCK_M * BLOCK_N;
#endif

    // Load convolution weights
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
        m_smem[m] = SM120_NEG_INF;
        l_smem[m] = 0.0f;
    }
    #pragma unroll 4
    for (int i = tid; i < BLOCK_M * D; i += num_threads) {
        acc_smem[i] = 0.0f;
    }
    __syncthreads();

    // Compute K/V iteration bounds
    int k_iter_start = 0;
    int k_iter_end = T_k;

    if (window_left >= 0) {
        int earliest_k = max(0, q_start - window_left - SM120_HALO_SIZE);
        k_iter_start = (earliest_k / BLOCK_N) * BLOCK_N;
    }

    if (IS_CAUSAL) {
        int latest_k = q_start + BLOCK_M - 1 + SM120_HALO_SIZE;
        k_iter_end = min(T_k, ((latest_k / BLOCK_N) + 1) * BLOCK_N);
    }

    // ============================================================
    // SYNCHRONOUS K/V LOADING (single-buffered for RTX 5090's 99KB limit)
    // ============================================================
    constexpr int K_STAGE_SIZE = (BLOCK_N + SM120_HALO_PADDED) * BLOCK_D;
    constexpr int V_STAGE_SIZE = BLOCK_N * BLOCK_D;

    // Calculate total number of K blocks
    const int num_k_blocks = (k_iter_end - k_iter_start + BLOCK_N - 1) / BLOCK_N;

    // ============================================================
    // MAIN LOOP over K/V blocks (synchronous loads)
    // ============================================================
    for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
        int k_block_start = k_iter_start + k_block_idx * BLOCK_N;

        // ============================================================
        // Load K with halo to shared memory (synchronous)
        // ============================================================
        #pragma unroll 4
        for (int i = tid; i < K_STAGE_SIZE; i += num_threads) {
            int n = i / BLOCK_D;
            int d = i % BLOCK_D;
            int k_pos = k_block_start - SM120_HALO_PADDED / 2 + n;

            if (k_pos >= 0 && k_pos < T_k && d < D) {
                K_smem[i] = K_batch[k_pos * D + d];
            } else {
                K_smem[i] = __float2bfloat16(0.0f);
            }
        }

        // ============================================================
        // Load V to shared memory (synchronous)
        // ============================================================
        #pragma unroll 4
        for (int i = tid; i < V_STAGE_SIZE; i += num_threads) {
            int n = i / BLOCK_D;
            int d = i % BLOCK_D;
            int v_pos = k_block_start + n;

            if (v_pos < T_k && d < D) {
                V_smem[i] = V_batch[v_pos * D + d];
            } else {
                V_smem[i] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        // Use K_smem and V_smem directly (no staging)
        __nv_bfloat16* K_smem_cur = K_smem;
        __nv_bfloat16* V_smem_cur = V_smem;

        // ============================================================
        // QK^T COMPUTATION using SM120 F8 MMA (16x8x32 tiles)
        // 2x throughput vs BF16 WMMA due to K=32 vs K=16
        // ============================================================
#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
        // F8 MMA tile configuration: 16x8x32
        // For QK^T: (BLOCK_M, BLOCK_D) x (BLOCK_D, S_STRIDE) -> (BLOCK_M, S_STRIDE)
        // Output tiles: (BLOCK_M/16) x (S_STRIDE/8) = 4 x 10 = 40 tiles for 64x80
        constexpr int F8_MMA_M = 16;
        constexpr int F8_MMA_N = 8;
        constexpr int F8_MMA_K = 32;
        constexpr int QK_M_TILES = BLOCK_M / F8_MMA_M;        // 4 for BLOCK_M=64
        constexpr int QK_N_TILES = S_STRIDE / F8_MMA_N;       // 10 for S_STRIDE=80
        constexpr int QK_K_ITERS = BLOCK_D / F8_MMA_K;        // 2 for BLOCK_D=64

        // Step 1: Quantize Q to E4M3 (once per K/V block since Q doesn't change)
        // Apply scale during quantization for better precision
        if (k_block_start == k_iter_start) {
            #pragma unroll 4
            for (int i = tid; i < BLOCK_M * BLOCK_D; i += num_threads) {
                int m = i / BLOCK_D;
                int d = i % BLOCK_D;
                float q_val = __bfloat162float(Q_smem[m * BLOCK_D + d]) * sm_scale;
                Q_e4m3[m * BLOCK_D + d] = float_to_e4m3(q_val);
            }
            __syncthreads();
        }

        // Step 2: Quantize K to E4M3 (per K/V block) - use current pipeline stage
        #pragma unroll 4
        for (int i = tid; i < (BLOCK_N + SM120_HALO_PADDED) * BLOCK_D; i += num_threads) {
            K_e4m3[i] = bf16_to_e4m3(K_smem_cur[i]);
        }
        __syncthreads();

        // Step 3: Compute QK^T using F8 MMA
        // Each warp handles multiple tiles
        const int total_qk_tiles = QK_M_TILES * QK_N_TILES;

        for (int tile_idx = warp_id; tile_idx < total_qk_tiles; tile_idx += num_warps) {
            int m_tile = tile_idx / QK_N_TILES;
            int n_tile = tile_idx % QK_N_TILES;

            // Initialize accumulators for this tile (4 floats per thread)
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

            // K dimension loop
            #pragma unroll
            for (int k_iter = 0; k_iter < QK_K_ITERS; k_iter++) {
                // Load A fragment (Q): 16 rows x 32 cols
                uint32_t a0, a1, a2, a3;
                const uint8_t* Q_tile = Q_e4m3 + m_tile * F8_MMA_M * BLOCK_D + k_iter * F8_MMA_K;
                load_a_fragment_f8(a0, a1, a2, a3, Q_tile, BLOCK_D, lane_id);

                // Load B fragment (K^T): 32 rows x 8 cols
                // K is (N_with_halo, D) row-major, need K^T which is (D, N_with_halo)
                uint32_t b0, b1;
                const uint8_t* K_tile = K_e4m3 + n_tile * F8_MMA_N * BLOCK_D + k_iter * F8_MMA_K;
                load_b_fragment_f8(b0, b1, K_tile, BLOCK_D, lane_id);

                // Execute F8 MMA
                mma_f8_16x8x32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
            }

            // Store result to S_smem
            // Note: scale was already applied during Q quantization
            float* S_tile = S_smem + m_tile * F8_MMA_M * S_STRIDE + n_tile * F8_MMA_N;
            store_d_fragment_f8(d0, d1, d2, d3, S_tile, S_STRIDE, lane_id);
        }
        __syncthreads();

#else  // Fallback to WMMA for non-F8 builds
        // WMMA QK^T (original implementation) - use current pipeline stage K
        for (int tile_idx = warp_id; tile_idx < M_TILES * N_TILES; tile_idx += num_warps) {
            int m_tile = tile_idx / N_TILES;
            int n_tile = tile_idx % N_TILES;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> q_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> k_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;

            fill_fragment(s_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
                load_matrix_sync(q_frag, Q_smem + m_tile * WMMA_M * BLOCK_D + k_tile * WMMA_K, BLOCK_D);
                load_matrix_sync(k_frag, K_smem_cur + n_tile * WMMA_N * BLOCK_D + k_tile * WMMA_K, BLOCK_D);
                mma_sync(s_frag, q_frag, k_frag, s_frag);
            }

            store_matrix_sync(S_smem + m_tile * WMMA_M * S_STRIDE + n_tile * WMMA_N,
                              s_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();
#endif

        // ============================================================
        // POST-MMA: APPLY SCALING AND MASKS
        // For F8 path, scale was applied during Q quantization
        // For WMMA path, scale is applied here
        // ============================================================
        #pragma unroll 1
        for (int i = tid; i < BLOCK_M * S_STRIDE; i += num_threads) {
            int m = i / S_STRIDE;
            int n_ext = i % S_STRIDE;
            int q_pos = q_start + m;
            int k_pos = k_block_start - SM120_HALO_PADDED / 2 + n_ext;

#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
            // F8 path: scale already applied, just load
            float s = S_smem[i];
#else
            // WMMA path: apply scale here
            float s = S_smem[i] * sm_scale;
#endif

            bool valid = (q_pos < T_q) && (k_pos >= 0) && (k_pos < T_k);
            if (!valid) s = SM120_NEG_INF;
            if (IS_CAUSAL && k_pos > q_pos) s = SM120_NEG_INF;
            if (window_left >= 0 && k_pos < q_pos - window_left) s = SM120_NEG_INF;

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
            float m_ij = SM120_NEG_INF;

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
        // PHASE 4: P @ V - accumulate directly to acc_smem
        // ============================================================

#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
        // F8 MMA tile configuration for P@V
        // P: (BLOCK_M, BLOCK_N) x V: (BLOCK_N, BLOCK_D) -> O: (BLOCK_M, BLOCK_D)
        // Output tiles: (BLOCK_M/16) x (BLOCK_D/8) = 4 x 8 = 32 tiles for 64x64
        constexpr int PV_F8_M_TILES = BLOCK_M / F8_MMA_M;     // 4 for BLOCK_M=64
        constexpr int PV_F8_N_TILES = BLOCK_D / F8_MMA_N;     // 8 for BLOCK_D=64
        constexpr int PV_F8_K_ITERS = BLOCK_N / F8_MMA_K;     // 2 for BLOCK_N=64

        // Step 1: Quantize P to E4M3 (softmax output, already in [0,1] range)
        #pragma unroll 4
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            P_e4m3[i] = bf16_to_e4m3(P_smem[i]);
        }
        __syncthreads();

        // Step 2: Quantize V to E4M3 - use current pipeline stage V
        #pragma unroll 4
        for (int i = tid; i < BLOCK_N * BLOCK_D; i += num_threads) {
            V_e4m3[i] = bf16_to_e4m3(V_smem_cur[i]);
        }
        __syncthreads();

        // Step 3: Compute P@V using F8 MMA, accumulate directly to acc_smem
        const int total_pv_tiles = PV_F8_M_TILES * PV_F8_N_TILES;

        for (int tile_idx = warp_id; tile_idx < total_pv_tiles; tile_idx += num_warps) {
            int m_tile = tile_idx / PV_F8_N_TILES;
            int n_tile = tile_idx % PV_F8_N_TILES;

            // Load current accumulator values
            float* acc_tile = acc_smem + m_tile * F8_MMA_M * D + n_tile * F8_MMA_N;
            const int row_group = lane_id / 4;
            const int col_base = (lane_id % 4) * 2;
            float d0 = acc_tile[row_group * D + col_base];
            float d1 = acc_tile[row_group * D + col_base + 1];
            float d2 = acc_tile[(row_group + 8) * D + col_base];
            float d3 = acc_tile[(row_group + 8) * D + col_base + 1];

            // K dimension loop (over BLOCK_N)
            #pragma unroll
            for (int k_iter = 0; k_iter < PV_F8_K_ITERS; k_iter++) {
                // Load A fragment (P): 16 rows x 32 cols
                uint32_t a0, a1, a2, a3;
                const uint8_t* P_tile = P_e4m3 + m_tile * F8_MMA_M * BLOCK_N + k_iter * F8_MMA_K;
                load_a_fragment_f8(a0, a1, a2, a3, P_tile, BLOCK_N, lane_id);

                // Load B fragment (V): 32 rows x 8 cols
                // V is (BLOCK_N, BLOCK_D) row-major, used directly (no transpose)
                uint32_t b0, b1;
                const uint8_t* V_tile = V_e4m3 + k_iter * F8_MMA_K * BLOCK_D + n_tile * F8_MMA_N;
                load_b_fragment_f8_direct(b0, b1, V_tile, BLOCK_D, lane_id);

                // Execute F8 MMA
                mma_f8_16x8x32(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, d0, d1, d2, d3);
            }

            // Store result back to acc_smem
            store_d_fragment_f8(d0, d1, d2, d3, acc_tile, D, lane_id);
        }
        __syncthreads();

#else  // Fallback to WMMA
        // WMMA P @ V - accumulate directly to acc_smem
        for (int tile_idx = warp_id; tile_idx < PV_M_TILES * PV_N_TILES; tile_idx += num_warps) {
            int m_tile = tile_idx / PV_N_TILES;
            int n_tile = tile_idx % PV_N_TILES;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> p_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> v_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;

            // Load current accumulator values
            load_matrix_sync(o_frag, acc_smem + m_tile * WMMA_M * D + n_tile * WMMA_N, D, mem_row_major);

            #pragma unroll
            for (int k_tile = 0; k_tile < PV_K_TILES; k_tile++) {
                load_matrix_sync(p_frag, P_smem + m_tile * WMMA_M * BLOCK_N + k_tile * WMMA_K, BLOCK_N);
                load_matrix_sync(v_frag, V_smem_cur + k_tile * WMMA_K * BLOCK_D + n_tile * WMMA_N, BLOCK_D);
                mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            store_matrix_sync(acc_smem + m_tile * WMMA_M * D + n_tile * WMMA_N,
                              o_frag, D, mem_row_major);
        }
        __syncthreads();
#endif
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

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel launcher implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

inline void launch_flash_look_around_fwd_sm120(
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

    // Launch configuration
    int num_q_blocks = (T_q + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_q_blocks, B * n_q_heads);
    dim3 block(128);  // 4 warps

    // Calculate shared memory size (single-buffered for RTX 5090's 99KB limit)
    // Note: Triple-buffering disabled to fit in shared memory
    size_t smem_size = 0;
    if (D <= 64) {
        constexpr int BLOCK_D = 64;
        constexpr int S_STRIDE = BLOCK_N + SM120_HALO_PADDED;

        // O_block_smem removed - accumulate directly to acc_smem
        smem_size = BLOCK_M * BLOCK_D * sizeof(__nv_bfloat16) +                     // Q_smem
                    (BLOCK_N + SM120_HALO_PADDED) * BLOCK_D * sizeof(__nv_bfloat16) +  // K_smem (single)
                    BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16) +                     // V_smem (single)
                    BLOCK_M * S_STRIDE * sizeof(float) +                            // S_smem
                    BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16) +                     // P_smem
                    BLOCK_M * sizeof(float) +                                       // m_smem
                    BLOCK_M * sizeof(float) +                                       // l_smem
                    BLOCK_M * BLOCK_D * sizeof(float) +                             // acc_smem
                    BLOCK_M * sizeof(float) * 2;                                    // m_block, l_block
                    // Note: w_shared[5] is static shared memory, not included here

#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
        // Add F8 MMA buffers when F8 path is enabled
        smem_size += BLOCK_M * BLOCK_D * sizeof(uint8_t) +                          // Q_e4m3
                     (BLOCK_N + SM120_HALO_PADDED) * BLOCK_D * sizeof(uint8_t) +    // K_e4m3
                     BLOCK_M * BLOCK_N * sizeof(uint8_t) +                          // P_e4m3
                     BLOCK_N * BLOCK_D * sizeof(uint8_t);                           // V_e4m3
#endif

        cudaError_t attr_err = cudaFuncSetAttribute(
            causal ? flash_look_around_fwd_kernel_sm120<BLOCK_M, BLOCK_N, BLOCK_D, true>
                   : flash_look_around_fwd_kernel_sm120<BLOCK_M, BLOCK_N, BLOCK_D, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        if (attr_err != cudaSuccess) {
            cudaGetLastError();  // Clear the error
            return;  // Caller will detect via cudaGetLastError
        }

        if (causal) {
            flash_look_around_fwd_kernel_sm120<BLOCK_M, BLOCK_N, BLOCK_D, true>
                <<<grid, block, smem_size, stream>>>(params);
        } else {
            flash_look_around_fwd_kernel_sm120<BLOCK_M, BLOCK_N, BLOCK_D, false>
                <<<grid, block, smem_size, stream>>>(params);
        }
    } else {
        // D = 128 case (single-buffered for RTX 5090's 99KB limit)
        // O_block_smem removed - accumulate directly to acc_smem
        constexpr int BLOCK_D = 128;
        constexpr int S_STRIDE = BLOCK_N + SM120_HALO_PADDED;

        smem_size = BLOCK_M * BLOCK_D * sizeof(__nv_bfloat16) +                     // Q_smem
                    (BLOCK_N + SM120_HALO_PADDED) * BLOCK_D * sizeof(__nv_bfloat16) +  // K_smem (single)
                    BLOCK_N * BLOCK_D * sizeof(__nv_bfloat16) +                     // V_smem (single)
                    BLOCK_M * S_STRIDE * sizeof(float) +                            // S_smem
                    BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16) +                     // P_smem
                    BLOCK_M * sizeof(float) +                                       // m_smem
                    BLOCK_M * sizeof(float) +                                       // l_smem
                    BLOCK_M * BLOCK_D * sizeof(float) +                             // acc_smem
                    BLOCK_M * sizeof(float) * 2;                                    // m_block, l_block
                    // Note: w_shared[5] is static shared memory, not included here

#if defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)
        // Add F8 MMA buffers when F8 path is enabled
        smem_size += BLOCK_M * BLOCK_D * sizeof(uint8_t) +                          // Q_e4m3
                     (BLOCK_N + SM120_HALO_PADDED) * BLOCK_D * sizeof(uint8_t) +    // K_e4m3
                     BLOCK_M * BLOCK_N * sizeof(uint8_t) +                          // P_e4m3
                     BLOCK_N * BLOCK_D * sizeof(uint8_t);                           // V_e4m3
#endif

        cudaError_t attr_err128 = cudaFuncSetAttribute(
            causal ? flash_look_around_fwd_kernel_sm120<BLOCK_M, BLOCK_N, BLOCK_D, true>
                   : flash_look_around_fwd_kernel_sm120<BLOCK_M, BLOCK_N, BLOCK_D, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        if (attr_err128 != cudaSuccess) {
            cudaGetLastError();  // Clear the error
            return;  // Caller will detect via cudaGetLastError
        }

        if (causal) {
            flash_look_around_fwd_kernel_sm120<BLOCK_M, BLOCK_N, BLOCK_D, true>
                <<<grid, block, smem_size, stream>>>(params);
        } else {
            flash_look_around_fwd_kernel_sm120<BLOCK_M, BLOCK_N, BLOCK_D, false>
                <<<grid, block, smem_size, stream>>>(params);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash_look_around
