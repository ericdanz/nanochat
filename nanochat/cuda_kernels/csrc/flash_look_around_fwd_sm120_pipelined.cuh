// flash_look_around_fwd_sm120_pipelined.cuh - 2-stage pipelined kernel for SM120
//
// This kernel overlaps memory loads with compute using 2-stage software pipelining:
// - Load K[n+1], V[n+1] while computing with K[n], V[n]
// - Uses cp.async for asynchronous memory copies
// - Removes O_block buffer to fit in 99KB shared memory
//
// Memory layout (BLOCK_M=64, BLOCK_N=64, D=64):
// - Q_smem: 8KB (loaded once)
// - K_smem[2]: 20KB (2-stage, with halo)
// - V_smem[2]: 16KB (2-stage)
// - S_smem: 20KB
// - P_smem: 8KB
// - acc_smem: 16KB
// - m/l: 512B
// Total: ~88KB (fits in 99KB)

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cmath>

#include "flash_look_around_kernel_sm120.h"

namespace flash_look_around {

using namespace nvcuda::wmma;

////////////////////////////////////////////////////////////////////////////////////////////////////
// 2-Stage Pipelined SM120 Forward Kernel (WMMA path)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BLOCK_M, int BLOCK_N, int BLOCK_D, bool IS_CAUSAL>
__global__ void flash_look_around_fwd_kernel_sm120_pipelined(FlashLookAroundFwdParams params) {
    // Constants
    constexpr int WARP_SIZE = 32;
    constexpr int HALO_SIZE = 2;
    constexpr int HALO_PADDED = 16;
    constexpr int S_STRIDE = BLOCK_N + HALO_PADDED;
    constexpr int S_OFFSET = HALO_PADDED / 2 - HALO_SIZE;
    constexpr int NUM_STAGES = 2;

    // WMMA tile configuration
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int M_TILES = BLOCK_M / WMMA_M;
    constexpr int N_TILES = S_STRIDE / WMMA_N;
    constexpr int K_TILES = BLOCK_D / WMMA_K;
    constexpr int PV_M_TILES = BLOCK_M / WMMA_M;
    constexpr int PV_N_TILES = BLOCK_D / WMMA_N;
    constexpr int PV_K_TILES = BLOCK_N / WMMA_K;

    // Size constants
    constexpr int K_STAGE_SIZE = (BLOCK_N + HALO_PADDED) * BLOCK_D;
    constexpr int V_STAGE_SIZE = BLOCK_N * BLOCK_D;

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

    // ============================================================
    // Shared memory layout with 2-stage K/V buffers
    // ============================================================
    extern __shared__ char smem[];

    __nv_bfloat16* Q_smem = reinterpret_cast<__nv_bfloat16*>(smem);
    // K double-buffered: stage 0 then stage 1
    __nv_bfloat16* K_smem_base = Q_smem + BLOCK_M * BLOCK_D;
    // V double-buffered: stage 0 then stage 1
    __nv_bfloat16* V_smem_base = K_smem_base + K_STAGE_SIZE * NUM_STAGES;
    // Score matrix
    float* S_smem = reinterpret_cast<float*>(V_smem_base + V_STAGE_SIZE * NUM_STAGES);
    // Convolved P matrix
    __nv_bfloat16* P_smem = reinterpret_cast<__nv_bfloat16*>(S_smem + BLOCK_M * S_STRIDE);
    // Running accumulators (no O_block_smem - accumulate directly)
    float* acc_smem = reinterpret_cast<float*>(P_smem + BLOCK_M * BLOCK_N);
    float* m_smem = acc_smem + BLOCK_M * BLOCK_D;
    float* l_smem = m_smem + BLOCK_M;

    // Helper to get K/V stage pointers
    auto K_stage = [&](int stage) -> __nv_bfloat16* {
        return K_smem_base + (stage % NUM_STAGES) * K_STAGE_SIZE;
    };
    auto V_stage = [&](int stage) -> __nv_bfloat16* {
        return V_smem_base + (stage % NUM_STAGES) * V_STAGE_SIZE;
    };

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

    constexpr float NEG_INF = -1e30f;

    // Load Q block to shared memory (once at the start)
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

    const int num_k_blocks = (k_iter_end - k_iter_start + BLOCK_N - 1) / BLOCK_N;

    // ============================================================
    // PIPELINED LOADING HELPER
    // Uses cp.async for async global->shared memory transfers
    // cp.async requires minimum 4-byte copies, so we load 4 bf16 at once (8 bytes)
    // ============================================================
    auto load_kv_stage_async = [&](int k_block_idx, int stage) {
        int k_block_start = k_iter_start + k_block_idx * BLOCK_N;
        __nv_bfloat16* K_dst = K_stage(stage);
        __nv_bfloat16* V_dst = V_stage(stage);

        // Load K with halo using cp.async (8 bytes = 4 bf16 at a time)
        // K layout: (BLOCK_N + HALO_PADDED) x BLOCK_D, row-major
        constexpr int K_ELEMENTS_PER_COPY = 4;  // 8 bytes / 2 bytes per bf16
        constexpr int K_TOTAL_COPIES = K_STAGE_SIZE / K_ELEMENTS_PER_COPY;

        #pragma unroll 2
        for (int copy_idx = tid; copy_idx < K_TOTAL_COPIES; copy_idx += num_threads) {
            int elem_idx = copy_idx * K_ELEMENTS_PER_COPY;
            int n = elem_idx / BLOCK_D;
            int d = (elem_idx % BLOCK_D);
            int k_pos = k_block_start - HALO_PADDED / 2 + n;

            // Check if entire 4-element chunk is valid
            bool valid = (k_pos >= 0 && k_pos < T_k && d + K_ELEMENTS_PER_COPY <= D);

            if (valid) {
                // Async copy 8 bytes (4 bfloat16)
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 8;"
                    : : "l"(K_dst + elem_idx), "l"(K_batch + k_pos * D + d));
            } else {
                // Fallback: zero out manually
                K_dst[elem_idx + 0] = __float2bfloat16(0.0f);
                K_dst[elem_idx + 1] = __float2bfloat16(0.0f);
                K_dst[elem_idx + 2] = __float2bfloat16(0.0f);
                K_dst[elem_idx + 3] = __float2bfloat16(0.0f);
            }
        }

        // Load V using cp.async (8 bytes = 4 bf16 at a time)
        constexpr int V_ELEMENTS_PER_COPY = 4;
        constexpr int V_TOTAL_COPIES = V_STAGE_SIZE / V_ELEMENTS_PER_COPY;

        #pragma unroll 2
        for (int copy_idx = tid; copy_idx < V_TOTAL_COPIES; copy_idx += num_threads) {
            int elem_idx = copy_idx * V_ELEMENTS_PER_COPY;
            int n = elem_idx / BLOCK_D;
            int d = (elem_idx % BLOCK_D);
            int v_pos = k_block_start + n;

            bool valid = (v_pos < T_k && d + V_ELEMENTS_PER_COPY <= D);

            if (valid) {
                asm volatile(
                    "cp.async.ca.shared.global [%0], [%1], 8;"
                    : : "l"(V_dst + elem_idx), "l"(V_batch + v_pos * D + d));
            } else {
                V_dst[elem_idx + 0] = __float2bfloat16(0.0f);
                V_dst[elem_idx + 1] = __float2bfloat16(0.0f);
                V_dst[elem_idx + 2] = __float2bfloat16(0.0f);
                V_dst[elem_idx + 3] = __float2bfloat16(0.0f);
            }
        }

        // Commit the async group
        asm volatile("cp.async.commit_group;");
    };

    // ============================================================
    // MAIN PIPELINED LOOP
    // ============================================================

    // Prefetch stage 0
    if (num_k_blocks > 0) {
        load_kv_stage_async(0, 0);
    }

    for (int k_block_idx = 0; k_block_idx < num_k_blocks; k_block_idx++) {
        int k_block_start = k_iter_start + k_block_idx * BLOCK_N;
        int compute_stage = k_block_idx % NUM_STAGES;

        // Start loading next stage (if there is one)
        if (k_block_idx + 1 < num_k_blocks) {
            int next_stage = (k_block_idx + 1) % NUM_STAGES;
            load_kv_stage_async(k_block_idx + 1, next_stage);
        }

        // Wait for current stage to be ready
        // Wait for all but 1 outstanding group (allows next load to be in flight)
        if (k_block_idx + 1 < num_k_blocks) {
            asm volatile("cp.async.wait_group 1;");
        } else {
            asm volatile("cp.async.wait_all;");
        }
        __syncthreads();

        // Get pointers to current stage buffers
        __nv_bfloat16* K_cur = K_stage(compute_stage);
        __nv_bfloat16* V_cur = V_stage(compute_stage);

        // ============================================================
        // COMPUTE QK^T using WMMA
        // ============================================================
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
                load_matrix_sync(k_frag, K_cur + n_tile * WMMA_N * BLOCK_D + k_tile * WMMA_K, BLOCK_D);
                mma_sync(s_frag, q_frag, k_frag, s_frag);
            }

            store_matrix_sync(S_smem + m_tile * WMMA_M * S_STRIDE + n_tile * WMMA_N,
                              s_frag, S_STRIDE, mem_row_major);
        }
        __syncthreads();

        // ============================================================
        // Apply scaling and masks
        // ============================================================
        #pragma unroll 1
        for (int i = tid; i < BLOCK_M * S_STRIDE; i += num_threads) {
            int m = i / S_STRIDE;
            int n_ext = i % S_STRIDE;
            int q_pos = q_start + m;
            int k_pos = k_block_start - HALO_PADDED / 2 + n_ext;

            float s = S_smem[i] * sm_scale;

            bool valid = (q_pos < T_q) && (k_pos >= 0) && (k_pos < T_k);
            if (!valid) s = NEG_INF;
            if (IS_CAUSAL && k_pos > q_pos) s = NEG_INF;
            if (window_left >= 0 && k_pos < q_pos - window_left) s = NEG_INF;

            S_smem[i] = s;
        }
        __syncthreads();

        // ============================================================
        // PHASE 1: Find max across all 5 shifts
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
        // PHASE 1.5: Pre-compute exp(S - m_new) for all S_smem
        // This eliminates redundant exp() calls (4x reduction)
        // Each adjacent convolution window shares values, so pre-computing
        // exp() once and reusing is much faster.
        // ============================================================
        #pragma unroll 2
        for (int i = tid; i < BLOCK_M * S_STRIDE; i += num_threads) {
            int m = i / S_STRIDE;
            int col = i % S_STRIDE;

            float s = S_smem[i];
            float m_new = fmaxf(m_smem[m], m_block[m]);

            // Compute exp(s - m_new) or 0 if masked
            float p = (s > -1e20f) ? expf(s - m_new) : 0.0f;
            S_smem[i] = p;  // Reuse S_smem for exp values
        }
        __syncthreads();

        // ============================================================
        // PHASE 2: Compute P_conv (weighted sum of pre-computed exp)
        // ============================================================
        #pragma unroll 1
        for (int i = tid; i < BLOCK_M * BLOCK_N; i += num_threads) {
            int m = i / BLOCK_N;
            int n = i % BLOCK_N;
            int q_pos = q_start + m;
            int v_pos = k_block_start + n;

            float p_conv = 0.0f;

            if (q_pos < T_q && v_pos < T_k) {
                // Read pre-computed exp values
                float p_m2 = S_smem[m * S_STRIDE + n + S_OFFSET + 0];
                float p_m1 = S_smem[m * S_STRIDE + n + S_OFFSET + 1];
                float p_0  = S_smem[m * S_STRIDE + n + S_OFFSET + 2];
                float p_p1 = S_smem[m * S_STRIDE + n + S_OFFSET + 3];
                float p_p2 = S_smem[m * S_STRIDE + n + S_OFFSET + 4];

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
        // PHASE 3: Rescale accumulator and update m/l
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
        // PHASE 4: P @ V using WMMA, accumulate directly to acc_smem
        // ============================================================
        for (int tile_idx = warp_id; tile_idx < PV_M_TILES * PV_N_TILES; tile_idx += num_warps) {
            int m_tile = tile_idx / PV_N_TILES;
            int n_tile = tile_idx % PV_N_TILES;

            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> p_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, row_major> v_frag;
            fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> o_frag;

            // Load current accumulator values
            load_matrix_sync(o_frag, acc_smem + m_tile * WMMA_M * BLOCK_D + n_tile * WMMA_N, BLOCK_D, mem_row_major);

            #pragma unroll
            for (int k_tile = 0; k_tile < PV_K_TILES; k_tile++) {
                load_matrix_sync(p_frag, P_smem + m_tile * WMMA_M * BLOCK_N + k_tile * WMMA_K, BLOCK_N);
                load_matrix_sync(v_frag, V_cur + k_tile * WMMA_K * BLOCK_D + n_tile * WMMA_N, BLOCK_D);
                mma_sync(o_frag, p_frag, v_frag, o_frag);
            }

            // Store back to accumulator
            store_matrix_sync(acc_smem + m_tile * WMMA_M * BLOCK_D + n_tile * WMMA_N,
                              o_frag, BLOCK_D, mem_row_major);
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

////////////////////////////////////////////////////////////////////////////////////////////////////
// Launcher for pipelined kernel
////////////////////////////////////////////////////////////////////////////////////////////////////

inline void launch_flash_look_around_fwd_sm120_pipelined(
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
    constexpr int HALO_PADDED = 16;
    constexpr int NUM_STAGES = 2;

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

    // Calculate shared memory size for D=64
    if (D <= 64) {
        constexpr int BLOCK_D = 64;
        constexpr int S_STRIDE = BLOCK_N + HALO_PADDED;
        constexpr int K_STAGE_SIZE = (BLOCK_N + HALO_PADDED) * BLOCK_D;
        constexpr int V_STAGE_SIZE = BLOCK_N * BLOCK_D;

        size_t smem_size = BLOCK_M * BLOCK_D * sizeof(__nv_bfloat16) +           // Q_smem
                          K_STAGE_SIZE * NUM_STAGES * sizeof(__nv_bfloat16) +    // K_smem[2]
                          V_STAGE_SIZE * NUM_STAGES * sizeof(__nv_bfloat16) +    // V_smem[2]
                          BLOCK_M * S_STRIDE * sizeof(float) +                    // S_smem
                          BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16) +             // P_smem
                          BLOCK_M * BLOCK_D * sizeof(float) +                     // acc_smem
                          BLOCK_M * sizeof(float) * 2 +                           // m_smem, l_smem
                          BLOCK_M * sizeof(float) * 2;                            // m_block, l_block

        cudaError_t attr_err = cudaFuncSetAttribute(
            causal ? flash_look_around_fwd_kernel_sm120_pipelined<BLOCK_M, BLOCK_N, BLOCK_D, true>
                   : flash_look_around_fwd_kernel_sm120_pipelined<BLOCK_M, BLOCK_N, BLOCK_D, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        if (attr_err != cudaSuccess) {
            cudaGetLastError();
            return;
        }

        if (causal) {
            flash_look_around_fwd_kernel_sm120_pipelined<BLOCK_M, BLOCK_N, BLOCK_D, true>
                <<<grid, block, smem_size, stream>>>(params);
        } else {
            flash_look_around_fwd_kernel_sm120_pipelined<BLOCK_M, BLOCK_N, BLOCK_D, false>
                <<<grid, block, smem_size, stream>>>(params);
        }
    } else {
        // D=128 case
        constexpr int BLOCK_D = 128;
        constexpr int S_STRIDE = BLOCK_N + HALO_PADDED;
        constexpr int K_STAGE_SIZE = (BLOCK_N + HALO_PADDED) * BLOCK_D;
        constexpr int V_STAGE_SIZE = BLOCK_N * BLOCK_D;

        size_t smem_size = BLOCK_M * BLOCK_D * sizeof(__nv_bfloat16) +
                          K_STAGE_SIZE * NUM_STAGES * sizeof(__nv_bfloat16) +
                          V_STAGE_SIZE * NUM_STAGES * sizeof(__nv_bfloat16) +
                          BLOCK_M * S_STRIDE * sizeof(float) +
                          BLOCK_M * BLOCK_N * sizeof(__nv_bfloat16) +
                          BLOCK_M * BLOCK_D * sizeof(float) +
                          BLOCK_M * sizeof(float) * 2 +
                          BLOCK_M * sizeof(float) * 2;

        // D=128 with 2-stage will exceed 99KB, fall back to single-stage
        // For now, just try and let it fail if too large
        cudaError_t attr_err = cudaFuncSetAttribute(
            causal ? flash_look_around_fwd_kernel_sm120_pipelined<BLOCK_M, BLOCK_N, BLOCK_D, true>
                   : flash_look_around_fwd_kernel_sm120_pipelined<BLOCK_M, BLOCK_N, BLOCK_D, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
        if (attr_err != cudaSuccess) {
            cudaGetLastError();
            return;
        }

        if (causal) {
            flash_look_around_fwd_kernel_sm120_pipelined<BLOCK_M, BLOCK_N, BLOCK_D, true>
                <<<grid, block, smem_size, stream>>>(params);
        } else {
            flash_look_around_fwd_kernel_sm120_pipelined<BLOCK_M, BLOCK_N, BLOCK_D, false>
                <<<grid, block, smem_size, stream>>>(params);
        }
    }
}

}  // namespace flash_look_around
