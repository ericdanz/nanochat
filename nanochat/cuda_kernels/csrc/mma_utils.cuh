// mma_utils.cuh - Tensor Core MMA (Matrix Multiply Accumulate) utilities
// Provides wrappers for wmma operations on bfloat16

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

namespace fused_look_around {

using namespace nvcuda;

// Tensor Core tile sizes for bf16
// m16n8k16 is efficient for Blackwell (sm_120)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Fragment types for bf16 matrix multiply
using FragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>;
using FragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>;
using FragBT = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major>;
using FragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

// Compute C = A @ B^T using Tensor Cores
// A: (M, K) row-major
// B: (N, K) row-major (will be transposed)
// C: (M, N) row-major
template <int M, int N, int K>
__device__ __forceinline__ void matmul_ABt(
    float* C,           // Output: (M, N)
    const __nv_bfloat16* A,  // (M, K)
    const __nv_bfloat16* B,  // (N, K)
    int lda,            // Leading dimension of A
    int ldb,            // Leading dimension of B
    int ldc             // Leading dimension of C
) {
    static_assert(M % WMMA_M == 0, "M must be divisible by WMMA_M");
    static_assert(N % WMMA_N == 0, "N must be divisible by WMMA_N");
    static_assert(K % WMMA_K == 0, "K must be divisible by WMMA_K");

    // Initialize output fragments
    FragC c_frag[M / WMMA_M][N / WMMA_N];
    #pragma unroll
    for (int mi = 0; mi < M / WMMA_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < N / WMMA_N; ni++) {
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);
        }
    }

    // Accumulate over K dimension
    #pragma unroll
    for (int ki = 0; ki < K / WMMA_K; ki++) {
        FragA a_frag[M / WMMA_M];
        FragBT b_frag[N / WMMA_N];

        // Load A fragments
        #pragma unroll
        for (int mi = 0; mi < M / WMMA_M; mi++) {
            wmma::load_matrix_sync(a_frag[mi], A + mi * WMMA_M * lda + ki * WMMA_K, lda);
        }

        // Load B fragments (B is row-major, we want B^T, so use row_major layout for B)
        #pragma unroll
        for (int ni = 0; ni < N / WMMA_N; ni++) {
            wmma::load_matrix_sync(b_frag[ni], B + ni * WMMA_N * ldb + ki * WMMA_K, ldb);
        }

        // Compute C += A @ B^T
        #pragma unroll
        for (int mi = 0; mi < M / WMMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < N / WMMA_N; ni++) {
                wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
            }
        }
    }

    // Store output fragments
    #pragma unroll
    for (int mi = 0; mi < M / WMMA_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < N / WMMA_N; ni++) {
            wmma::store_matrix_sync(C + mi * WMMA_M * ldc + ni * WMMA_N, c_frag[mi][ni], ldc, wmma::mem_row_major);
        }
    }
}

// Compute C = A @ B using Tensor Cores
// A: (M, K) row-major
// B: (K, N) col-major (or equivalently (N, K) row-major transposed)
// C: (M, N) row-major
template <int M, int N, int K>
__device__ __forceinline__ void matmul_AB(
    float* C,
    const __nv_bfloat16* A,  // (M, K)
    const __nv_bfloat16* B,  // (K, N)
    int lda,
    int ldb,
    int ldc
) {
    static_assert(M % WMMA_M == 0, "M must be divisible by WMMA_M");
    static_assert(N % WMMA_N == 0, "N must be divisible by WMMA_N");
    static_assert(K % WMMA_K == 0, "K must be divisible by WMMA_K");

    FragC c_frag[M / WMMA_M][N / WMMA_N];
    #pragma unroll
    for (int mi = 0; mi < M / WMMA_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < N / WMMA_N; ni++) {
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);
        }
    }

    #pragma unroll
    for (int ki = 0; ki < K / WMMA_K; ki++) {
        FragA a_frag[M / WMMA_M];
        FragB b_frag[N / WMMA_N];

        #pragma unroll
        for (int mi = 0; mi < M / WMMA_M; mi++) {
            wmma::load_matrix_sync(a_frag[mi], A + mi * WMMA_M * lda + ki * WMMA_K, lda);
        }

        #pragma unroll
        for (int ni = 0; ni < N / WMMA_N; ni++) {
            wmma::load_matrix_sync(b_frag[ni], B + ki * WMMA_K + ni * WMMA_N * ldb, ldb);
        }

        #pragma unroll
        for (int mi = 0; mi < M / WMMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < N / WMMA_N; ni++) {
                wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
            }
        }
    }

    #pragma unroll
    for (int mi = 0; mi < M / WMMA_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < N / WMMA_N; ni++) {
            wmma::store_matrix_sync(C + mi * WMMA_M * ldc + ni * WMMA_N, c_frag[mi][ni], ldc, wmma::mem_row_major);
        }
    }
}

// Compute C = A^T @ B using Tensor Cores
// A: (K, M) row-major (will be transposed to get (M, K))
// B: (K, N) row-major
// C: (M, N) row-major
template <int M, int N, int K>
__device__ __forceinline__ void matmul_AtB(
    float* C,
    const __nv_bfloat16* A,  // (K, M) row-major
    const __nv_bfloat16* B,  // (K, N) row-major
    int lda,
    int ldb,
    int ldc
) {
    static_assert(M % WMMA_M == 0, "M must be divisible by WMMA_M");
    static_assert(N % WMMA_N == 0, "N must be divisible by WMMA_N");
    static_assert(K % WMMA_K == 0, "K must be divisible by WMMA_K");

    // For A^T, we load A with col_major layout
    using FragAT = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major>;

    FragC c_frag[M / WMMA_M][N / WMMA_N];
    #pragma unroll
    for (int mi = 0; mi < M / WMMA_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < N / WMMA_N; ni++) {
            wmma::fill_fragment(c_frag[mi][ni], 0.0f);
        }
    }

    #pragma unroll
    for (int ki = 0; ki < K / WMMA_K; ki++) {
        FragAT a_frag[M / WMMA_M];
        FragB b_frag[N / WMMA_N];

        // Load A^T: A is (K, M) row-major, load as col-major to get A^T
        #pragma unroll
        for (int mi = 0; mi < M / WMMA_M; mi++) {
            wmma::load_matrix_sync(a_frag[mi], A + ki * WMMA_K * lda + mi * WMMA_M, lda);
        }

        #pragma unroll
        for (int ni = 0; ni < N / WMMA_N; ni++) {
            wmma::load_matrix_sync(b_frag[ni], B + ki * WMMA_K * ldb + ni * WMMA_N, ldb);
        }

        #pragma unroll
        for (int mi = 0; mi < M / WMMA_M; mi++) {
            #pragma unroll
            for (int ni = 0; ni < N / WMMA_N; ni++) {
                wmma::mma_sync(c_frag[mi][ni], a_frag[mi], b_frag[ni], c_frag[mi][ni]);
            }
        }
    }

    #pragma unroll
    for (int mi = 0; mi < M / WMMA_M; mi++) {
        #pragma unroll
        for (int ni = 0; ni < N / WMMA_N; ni++) {
            wmma::store_matrix_sync(C + mi * WMMA_M * ldc + ni * WMMA_N, c_frag[mi][ni], ldc, wmma::mem_row_major);
        }
    }
}

// Simple dot product using warp-level reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction for computing row sums
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem_scratch) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // Warp reduction
    val = warp_reduce_sum(val);

    // Write warp results to shared memory
    if (lane == 0) {
        smem_scratch[warp_id] = val;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0) {
        val = (threadIdx.x < BLOCK_SIZE / 32) ? smem_scratch[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }

    return val;
}

// Max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_max(float val, float* smem_scratch) {
    int lane = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        smem_scratch[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (threadIdx.x < BLOCK_SIZE / 32) ? smem_scratch[lane] : -INFINITY;
        val = warp_reduce_max(val);
    }

    return val;
}

}  // namespace fused_look_around
