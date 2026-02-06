// flash_look_around_fwd_sm120_cute.cuh - Hybrid CuTe + Custom Softmax/Conv Kernel
//
// Architecture:
// - CuTe TiledMMA for QK^T and P@V (tcgen05.mma, handles layouts automatically)
// - TMEM for MMA accumulators (frees SMEM, reduces bank conflicts)
// - Custom 5-tap convolution + online softmax in SMEM
//
// This combines CUTLASS/CuTe's optimized tensor core utilization with our
// custom look-around convolution logic.

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

// CuTe includes
#include <cute/tensor.hpp>
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/mma_sm100.hpp>
#include <cute/arch/mma_sm100_umma.hpp>
// TMEM disabled for now - requires CUTE_ARCH_TCGEN05_TMEM_ENABLED
// #include <cute/arch/tmem_allocator_sm100.hpp>

// CUTLASS includes
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/bfloat16.h>

#include "flash_look_around_kernel_sm120.h"

namespace flash_look_around {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Configuration
////////////////////////////////////////////////////////////////////////////////////////////////////

// Tile sizes optimized for look-around attention with 5-tap convolution
struct LookAroundConfig {
    static constexpr int BLOCK_M = 64;      // Query tile size
    static constexpr int BLOCK_N = 64;      // Key/Value tile size
    static constexpr int BLOCK_D = 64;      // Head dimension
    static constexpr int HALO = 2;          // ±2 for 5-tap convolution
    static constexpr int HALO_PADDED = 16;  // Padded for alignment
    static constexpr int S_STRIDE = BLOCK_N + HALO_PADDED;  // Score matrix width with halo

    // For SM100 tcgen05.mma with BF16
    // The smallest MMA is 64x64x16 for BF16 (from cute/arch/mma_sm100.hpp)
    static constexpr int MMA_M = 64;
    static constexpr int MMA_N = 64;
    static constexpr int MMA_K = 16;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shared Storage Layout
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class Element, class Config = LookAroundConfig>
struct LookAroundSharedStorage {
    // Q: Loaded once, reused for all K blocks
    // Layout: (BLOCK_M, BLOCK_D) K-major for MMA A operand
    alignas(128) Element smem_Q[Config::BLOCK_M * Config::BLOCK_D];

    // K with halo: Double-buffered for pipelining
    // Layout: (S_STRIDE, BLOCK_D) K-major for MMA B operand
    alignas(128) Element smem_K[2][Config::S_STRIDE * Config::BLOCK_D];

    // V: Double-buffered for pipelining
    // Layout: (BLOCK_N, BLOCK_D) K-major for MMA B operand
    alignas(128) Element smem_V[2][Config::BLOCK_N * Config::BLOCK_D];

    // S scores: Materialized from TMEM for convolution
    // After QK^T: S[m,n] = Q[m,:] @ K[n,:]^T
    alignas(128) float smem_S[Config::BLOCK_M * Config::S_STRIDE];

    // P (convolved softmax output): Input for P@V MMA
    alignas(128) Element smem_P[Config::BLOCK_M * Config::BLOCK_N];

    // Online softmax accumulators
    alignas(64) float smem_m[Config::BLOCK_M];  // Running max
    alignas(64) float smem_l[Config::BLOCK_M];  // Running sum

    // Output accumulator (rescaled each iteration)
    alignas(128) float smem_O[Config::BLOCK_M * Config::BLOCK_D];

    // Convolution weights
    alignas(16) float conv_weights[5];

    // Barriers for MMA synchronization
    alignas(16) uint64_t mma_barrier_qk;
    alignas(16) uint64_t mma_barrier_pv;

    // TMEM base pointer (allocated by warp 0)
    alignas(16) uint32_t tmem_base;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel Implementation
////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool IS_CAUSAL>
__global__ void __launch_bounds__(128, 1)
flash_look_around_fwd_kernel_cute(FlashLookAroundFwdParams params) {

    using Config = LookAroundConfig;
    using Element = __nv_bfloat16;
    using CutlassElement = cutlass::bfloat16_t;
    using Storage = LookAroundSharedStorage<Element, Config>;

    // Thread/warp indices
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int num_threads = blockDim.x;
    const uint32_t elect_one = cute::elect_one_sync();
    const bool is_warp0 = (warp_id == 0);

    // Block indices
    const int q_block_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    const int batch_idx = bh_idx / params.n_q_heads;
    const int q_head_idx = bh_idx % params.n_q_heads;

    // GQA mapping
    const int heads_per_kv = params.n_q_heads / params.n_kv_heads;
    const int kv_head_idx = q_head_idx / heads_per_kv;

    // Query range
    const int q_start = q_block_idx * Config::BLOCK_M;
    const int T_q = params.T_q;
    const int T_k = params.T_k;
    const int D = params.D;

    // Input pointers
    const Element* Q_batch = reinterpret_cast<const Element*>(params.Q_ptr)
        + batch_idx * params.Q_batch_stride + q_head_idx * params.Q_head_stride;
    const Element* K_batch = reinterpret_cast<const Element*>(params.K_ptr)
        + batch_idx * params.K_batch_stride + kv_head_idx * params.K_head_stride;
    const Element* V_batch = reinterpret_cast<const Element*>(params.V_ptr)
        + batch_idx * params.V_batch_stride + kv_head_idx * params.V_head_stride;
    const float* conv_weights_ptr = reinterpret_cast<const float*>(params.conv_weights_ptr)
        + kv_head_idx * 5;
    Element* O_batch = reinterpret_cast<Element*>(params.O_ptr)
        + batch_idx * params.O_batch_stride + q_head_idx * params.O_head_stride;
    float* LSE_batch = params.LSE_ptr
        + batch_idx * params.n_q_heads * T_q + q_head_idx * T_q;

    // Shared memory
    extern __shared__ char smem_raw[];
    Storage& storage = *reinterpret_cast<Storage*>(smem_raw);

    // Load convolution weights
    if (tid < 5) {
        storage.conv_weights[tid] = conv_weights_ptr[tid];
    }

    // Initialize barriers (warp 0, lane 0)
    if (is_warp0 && elect_one) {
        cute::initialize_barrier(storage.mma_barrier_qk, 1);
        cute::initialize_barrier(storage.mma_barrier_pv, 1);
    }

    // ========================================================================
    // TMEM Allocation - DISABLED for now (requires CUTE_ARCH_TCGEN05_TMEM_ENABLED)
    // When using CuTe TiledMMA, we'll enable this for TMEM accumulators
    // ========================================================================
    // using TmemAllocator = cute::TMEM::Allocator1Sm;
    // TmemAllocator tmem_allocator{};
    // if (is_warp0 && elect_one) {
    //     tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, &storage.tmem_base);
    // }
    __syncthreads();

    const float sm_scale = params.sm_scale;
    const float w0 = storage.conv_weights[0];
    const float w1 = storage.conv_weights[1];
    const float w2 = storage.conv_weights[2];
    const float w3 = storage.conv_weights[3];
    const float w4 = storage.conv_weights[4];

    constexpr float NEG_INF = -1e30f;
    constexpr int S_OFFSET = Config::HALO_PADDED / 2 - Config::HALO;

    // ========================================================================
    // Create CuTe TiledMMA for QK^T
    // QK^T: Q (M x D) @ K^T (D x N+halo) -> S (M x N+halo)
    // A = Q is M x K (K-major = row-major), B = K is N x K (K-major = col-major for K^T)
    // ========================================================================

    // For QK^T with 64x80x64 tiles (80 = 64 + 16 halo)
    // We use 64x64x16 MMA atoms
    // Note: SM100_MMA_F16BF16_SS uses BF16 inputs, F32 accumulator
    TiledMMA tiled_mma_qk = make_tiled_mma(
        SM100_MMA_F16BF16_SS<CutlassElement, CutlassElement, float,
                            64, 64,  // MMA M, N dimensions
                            UMMA::Major::K, UMMA::Major::K>{}  // Both A and B are K-major
    );

    // For P@V with 64x64x64 tiles
    TiledMMA tiled_mma_pv = make_tiled_mma(
        SM100_MMA_F16BF16_SS<CutlassElement, CutlassElement, float,
                            64, 64,
                            UMMA::Major::K, UMMA::Major::K>{}
    );

    // ========================================================================
    // Load Q to SMEM (once at start)
    // ========================================================================
    #pragma unroll 4
    for (int i = tid; i < Config::BLOCK_M * Config::BLOCK_D; i += num_threads) {
        int m = i / Config::BLOCK_D;
        int d = i % Config::BLOCK_D;
        int q_pos = q_start + m;
        if (q_pos < T_q && d < D) {
            storage.smem_Q[i] = Q_batch[q_pos * D + d];
        } else {
            storage.smem_Q[i] = __float2bfloat16(0.0f);
        }
    }

    // Initialize online softmax state
    #pragma unroll
    for (int m = tid; m < Config::BLOCK_M; m += num_threads) {
        storage.smem_m[m] = NEG_INF;
        storage.smem_l[m] = 0.0f;
    }
    #pragma unroll 4
    for (int i = tid; i < Config::BLOCK_M * D; i += num_threads) {
        storage.smem_O[i] = 0.0f;
    }
    __syncthreads();

    // ========================================================================
    // Compute K/V iteration bounds
    // ========================================================================
    int k_iter_start = 0;
    int k_iter_end = T_k;

    if (params.window_left >= 0) {
        int earliest_k = max(0, q_start - params.window_left - Config::HALO);
        k_iter_start = (earliest_k / Config::BLOCK_N) * Config::BLOCK_N;
    }

    if (IS_CAUSAL) {
        int latest_k = q_start + Config::BLOCK_M - 1 + Config::HALO;
        k_iter_end = min(T_k, ((latest_k / Config::BLOCK_N) + 1) * Config::BLOCK_N);
    }

    const int num_k_blocks = (k_iter_end - k_iter_start + Config::BLOCK_N - 1) / Config::BLOCK_N;

    // ========================================================================
    // Main Loop: Process K/V blocks
    // ========================================================================

    // Prefetch first K/V block (stage 0)
    if (num_k_blocks > 0) {
        int k_block_start = k_iter_start;

        // Load K with halo
        #pragma unroll 2
        for (int i = tid; i < Config::S_STRIDE * Config::BLOCK_D; i += num_threads) {
            int n = i / Config::BLOCK_D;
            int d = i % Config::BLOCK_D;
            int k_pos = k_block_start - Config::HALO_PADDED / 2 + n;

            if (k_pos >= 0 && k_pos < T_k && d < D) {
                storage.smem_K[0][i] = K_batch[k_pos * D + d];
            } else {
                storage.smem_K[0][i] = __float2bfloat16(0.0f);
            }
        }

        // Load V
        #pragma unroll 2
        for (int i = tid; i < Config::BLOCK_N * Config::BLOCK_D; i += num_threads) {
            int n = i / Config::BLOCK_D;
            int d = i % Config::BLOCK_D;
            int v_pos = k_block_start + n;

            if (v_pos < T_k && d < D) {
                storage.smem_V[0][i] = V_batch[v_pos * D + d];
            } else {
                storage.smem_V[0][i] = __float2bfloat16(0.0f);
            }
        }
    }
    __syncthreads();

    int mma_barrier_phase_qk = 0;
    int mma_barrier_phase_pv = 0;

    for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
        int k_block_start = k_iter_start + k_block_idx * Config::BLOCK_N;
        int compute_stage = k_block_idx % 2;
        int next_stage = (k_block_idx + 1) % 2;

        // ====================================================================
        // Async load next K/V block (pipelining)
        // ====================================================================
        if (k_block_idx + 1 < num_k_blocks) {
            int next_k_start = k_iter_start + (k_block_idx + 1) * Config::BLOCK_N;

            #pragma unroll 2
            for (int i = tid; i < Config::S_STRIDE * Config::BLOCK_D; i += num_threads) {
                int n = i / Config::BLOCK_D;
                int d = i % Config::BLOCK_D;
                int k_pos = next_k_start - Config::HALO_PADDED / 2 + n;

                if (k_pos >= 0 && k_pos < T_k && d < D) {
                    storage.smem_K[next_stage][i] = K_batch[k_pos * D + d];
                } else {
                    storage.smem_K[next_stage][i] = __float2bfloat16(0.0f);
                }
            }

            #pragma unroll 2
            for (int i = tid; i < Config::BLOCK_N * Config::BLOCK_D; i += num_threads) {
                int n = i / Config::BLOCK_D;
                int d = i % Config::BLOCK_D;
                int v_pos = next_k_start + n;

                if (v_pos < T_k && d < D) {
                    storage.smem_V[next_stage][i] = V_batch[v_pos * D + d];
                } else {
                    storage.smem_V[next_stage][i] = __float2bfloat16(0.0f);
                }
            }
        }

        // ====================================================================
        // PHASE 1: QK^T using WMMA (fallback - CuTe TiledMMA requires more setup)
        // TODO: Replace with CuTe TiledMMA once SMEM layouts are properly configured
        // ====================================================================

        // Get pointers to current stage
        Element* K_cur = storage.smem_K[compute_stage];
        Element* V_cur = storage.smem_V[compute_stage];

        // WMMA QK^T computation (working fallback)
        using namespace nvcuda::wmma;
        constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
        constexpr int M_TILES = Config::BLOCK_M / WMMA_M;
        constexpr int N_TILES = Config::S_STRIDE / WMMA_N;
        constexpr int K_TILES = Config::BLOCK_D / WMMA_K;

        const int num_warps = num_threads / 32;

        for (int tile_idx = warp_id; tile_idx < M_TILES * N_TILES; tile_idx += num_warps) {
            int m_tile = tile_idx / N_TILES;
            int n_tile = tile_idx % N_TILES;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> q_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, col_major> k_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> s_frag;

            fill_fragment(s_frag, 0.0f);

            #pragma unroll
            for (int k_tile = 0; k_tile < K_TILES; k_tile++) {
                load_matrix_sync(q_frag, storage.smem_Q + m_tile * WMMA_M * Config::BLOCK_D + k_tile * WMMA_K, Config::BLOCK_D);
                load_matrix_sync(k_frag, K_cur + n_tile * WMMA_N * Config::BLOCK_D + k_tile * WMMA_K, Config::BLOCK_D);
                mma_sync(s_frag, q_frag, k_frag, s_frag);
            }

            store_matrix_sync(storage.smem_S + m_tile * WMMA_M * Config::S_STRIDE + n_tile * WMMA_N,
                             s_frag, Config::S_STRIDE, mem_row_major);
        }
        __syncthreads();

        // ====================================================================
        // Apply scaling and masks
        // ====================================================================
        #pragma unroll 1
        for (int i = tid; i < Config::BLOCK_M * Config::S_STRIDE; i += num_threads) {
            int m = i / Config::S_STRIDE;
            int n_ext = i % Config::S_STRIDE;
            int q_pos = q_start + m;
            int k_pos = k_block_start - Config::HALO_PADDED / 2 + n_ext;

            float s = storage.smem_S[i] * sm_scale;

            bool valid = (q_pos < T_q) && (k_pos >= 0) && (k_pos < T_k);
            if (!valid) s = NEG_INF;
            if (IS_CAUSAL && k_pos > q_pos) s = NEG_INF;
            if (params.window_left >= 0 && k_pos < q_pos - params.window_left) s = NEG_INF;

            storage.smem_S[i] = s;
        }
        __syncthreads();

        // ====================================================================
        // PHASE 2: Find max across all 5 shifts
        // ====================================================================
        __shared__ float m_block[Config::BLOCK_M];
        __shared__ float l_block[Config::BLOCK_M];

        #pragma unroll 1
        for (int m = tid; m < Config::BLOCK_M; m += num_threads) {
            int q_pos = q_start + m;
            float m_ij = NEG_INF;

            if (q_pos < T_q) {
                #pragma unroll 4
                for (int n = 0; n < Config::BLOCK_N; n++) {
                    int v_pos = k_block_start + n;
                    if (v_pos >= T_k) continue;

                    float s_m2 = storage.smem_S[m * Config::S_STRIDE + n + S_OFFSET + 0];
                    float s_m1 = storage.smem_S[m * Config::S_STRIDE + n + S_OFFSET + 1];
                    float s_0  = storage.smem_S[m * Config::S_STRIDE + n + S_OFFSET + 2];
                    float s_p1 = storage.smem_S[m * Config::S_STRIDE + n + S_OFFSET + 3];
                    float s_p2 = storage.smem_S[m * Config::S_STRIDE + n + S_OFFSET + 4];

                    m_ij = fmaxf(m_ij, fmaxf(fmaxf(s_m2, s_m1), fmaxf(fmaxf(s_0, s_p1), s_p2)));
                }
            }
            m_block[m] = m_ij;
        }
        __syncthreads();

        // ====================================================================
        // PHASE 3: Pre-compute exp(S - m_new) for convolution
        // ====================================================================
        #pragma unroll 2
        for (int i = tid; i < Config::BLOCK_M * Config::S_STRIDE; i += num_threads) {
            int m = i / Config::S_STRIDE;
            float s = storage.smem_S[i];
            float m_new = fmaxf(storage.smem_m[m], m_block[m]);
            float p = (s > -1e20f) ? expf(s - m_new) : 0.0f;
            storage.smem_S[i] = p;  // Reuse S for exp values
        }
        __syncthreads();

        // ====================================================================
        // PHASE 4: Compute P_conv (5-tap convolution on exp values)
        // ====================================================================
        #pragma unroll 1
        for (int i = tid; i < Config::BLOCK_M * Config::BLOCK_N; i += num_threads) {
            int m = i / Config::BLOCK_N;
            int n = i % Config::BLOCK_N;
            int q_pos = q_start + m;
            int v_pos = k_block_start + n;

            float p_conv = 0.0f;

            if (q_pos < T_q && v_pos < T_k) {
                float p_m2 = storage.smem_S[m * Config::S_STRIDE + n + S_OFFSET + 0];
                float p_m1 = storage.smem_S[m * Config::S_STRIDE + n + S_OFFSET + 1];
                float p_0  = storage.smem_S[m * Config::S_STRIDE + n + S_OFFSET + 2];
                float p_p1 = storage.smem_S[m * Config::S_STRIDE + n + S_OFFSET + 3];
                float p_p2 = storage.smem_S[m * Config::S_STRIDE + n + S_OFFSET + 4];

                p_conv = w0 * p_p2 + w1 * p_p1 + w2 * p_0 + w3 * p_m1 + w4 * p_m2;

                if (IS_CAUSAL && v_pos > q_pos) {
                    p_conv = 0.0f;
                }
            }

            storage.smem_P[i] = __float2bfloat16(p_conv);
        }
        __syncthreads();

        // Compute row sums
        #pragma unroll 1
        for (int m = tid; m < Config::BLOCK_M; m += num_threads) {
            float l_ij = 0.0f;
            #pragma unroll 4
            for (int n = 0; n < Config::BLOCK_N; n++) {
                l_ij += __bfloat162float(storage.smem_P[m * Config::BLOCK_N + n]);
            }
            l_block[m] = l_ij;
        }
        __syncthreads();

        // ====================================================================
        // PHASE 5: Rescale accumulator
        // ====================================================================
        #pragma unroll 1
        for (int m = tid; m < Config::BLOCK_M; m += num_threads) {
            int q_pos = q_start + m;
            if (q_pos >= T_q) continue;

            float m_i = storage.smem_m[m];
            float l_i = storage.smem_l[m];
            float m_new = fmaxf(m_i, m_block[m]);
            float alpha = expf(m_i - m_new);

            float* O_row = storage.smem_O + m * D;
            #pragma unroll 8
            for (int d = 0; d < D; d++) {
                O_row[d] *= alpha;
            }

            storage.smem_m[m] = m_new;
            storage.smem_l[m] = l_i * alpha + l_block[m];
        }
        __syncthreads();

        // ====================================================================
        // PHASE 6: P @ V using WMMA
        // ====================================================================
        constexpr int PV_M_TILES = Config::BLOCK_M / WMMA_M;
        constexpr int PV_N_TILES = Config::BLOCK_D / WMMA_N;
        constexpr int PV_K_TILES = Config::BLOCK_N / WMMA_K;

        for (int tile_idx = warp_id; tile_idx < PV_M_TILES * PV_N_TILES; tile_idx += num_warps) {
            int m_tile = tile_idx / PV_N_TILES;
            int n_tile = tile_idx % PV_N_TILES;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> p_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> v_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;

            // Load current accumulator
            load_matrix_sync(o_frag, storage.smem_O + m_tile * WMMA_M * Config::BLOCK_D + n_tile * WMMA_N, Config::BLOCK_D, mem_row_major);

            #pragma unroll
            for (int k_tile = 0; k_tile < PV_K_TILES; k_tile++) {
                load_matrix_sync(p_frag, storage.smem_P + m_tile * WMMA_M * Config::BLOCK_N + k_tile * WMMA_K, Config::BLOCK_N);
                load_matrix_sync(v_frag, V_cur + k_tile * WMMA_K * Config::BLOCK_D + n_tile * WMMA_N, Config::BLOCK_D);
                mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            store_matrix_sync(storage.smem_O + m_tile * WMMA_M * Config::BLOCK_D + n_tile * WMMA_N,
                             o_frag, Config::BLOCK_D, mem_row_major);
        }
        __syncthreads();
    }

    // ========================================================================
    // Write Output
    // ========================================================================
    #pragma unroll 1
    for (int m = tid; m < Config::BLOCK_M; m += num_threads) {
        int q_pos = q_start + m;
        if (q_pos >= T_q) continue;

        float l_final = fmaxf(storage.smem_l[m], 1e-9f);
        float m_final = storage.smem_m[m];
        float inv_l = 1.0f / l_final;

        const float* O_row = storage.smem_O + m * D;
        Element* out_row = O_batch + q_pos * D;

        #pragma unroll 8
        for (int d = 0; d < D; d++) {
            out_row[d] = __float2bfloat16(O_row[d] * inv_l);
        }

        LSE_batch[q_pos] = m_final + logf(l_final);
    }

    // ========================================================================
    // Cleanup: Free TMEM - DISABLED (not using TMEM yet)
    // ========================================================================
    // __syncthreads();
    // if (is_warp0 && elect_one) {
    //     tmem_allocator.release_allocation_lock();
    //     tmem_allocator.free(storage.tmem_base, TmemAllocator::Sm100TmemCapacityColumns);
    // }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launcher
////////////////////////////////////////////////////////////////////////////////////////////////////

inline void launch_flash_look_around_fwd_sm120_cute(
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
    using Config = LookAroundConfig;

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

    // Strides
    params.Q_batch_stride = n_q_heads * T_q * D;
    params.Q_head_stride = T_q * D;
    params.K_batch_stride = n_kv_heads * T_k * D;
    params.K_head_stride = T_k * D;
    params.V_batch_stride = n_kv_heads * T_k * D;
    params.V_head_stride = T_k * D;
    params.O_batch_stride = n_q_heads * T_q * D;
    params.O_head_stride = T_q * D;

    // Grid/block config
    int num_q_blocks = (T_q + Config::BLOCK_M - 1) / Config::BLOCK_M;
    dim3 grid(num_q_blocks, B * n_q_heads);
    dim3 block(128);  // 4 warps

    // Shared memory size
    size_t smem_size = sizeof(LookAroundSharedStorage<__nv_bfloat16, Config>);

    cudaFuncSetAttribute(
        causal ? flash_look_around_fwd_kernel_cute<true>
               : flash_look_around_fwd_kernel_cute<false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    );

    if (causal) {
        flash_look_around_fwd_kernel_cute<true><<<grid, block, smem_size, stream>>>(params);
    } else {
        flash_look_around_fwd_kernel_cute<false><<<grid, block, smem_size, stream>>>(params);
    }
}

}  // namespace flash_look_around
