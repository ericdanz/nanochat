// async_copy.cuh - Asynchronous copy utilities for CUDA kernels
// Provides cp.async wrappers for efficient global->shared memory transfers

#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace fused_look_around {

// cp.async for 16-byte (128-bit) transfers - optimal for bfloat16 x 8
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr, bool predicate) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %3, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 16;\n"
        "}\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(16), "r"((int)predicate)
    );
}

// cp.async for 8-byte (64-bit) transfers
__device__ __forceinline__ void cp_async_cg_8(void* smem_ptr, const void* gmem_ptr, bool predicate) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %3, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 8;\n"
        "}\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(8), "r"((int)predicate)
    );
}

// cp.async for 4-byte (32-bit) transfers
__device__ __forceinline__ void cp_async_cg_4(void* smem_ptr, const void* gmem_ptr, bool predicate) {
    unsigned smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  setp.ne.b32 p, %3, 0;\n"
        "  @p cp.async.cg.shared.global [%0], [%1], 4;\n"
        "}\n"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(4), "r"((int)predicate)
    );
}

// Commit all pending cp.async operations
__device__ __forceinline__ void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait for all cp.async operations to complete
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

// Wait for all but N groups
template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Load a row of bf16 values from global to shared memory using cp.async
// Assumes D is divisible by 8 for optimal 16-byte transfers
template <int D>
__device__ __forceinline__ void load_row_async(
    __nv_bfloat16* smem_row,
    const __nv_bfloat16* gmem_row,
    bool valid
) {
    #pragma unroll
    for (int d = 0; d < D; d += 8) {
        cp_async_cg_16(smem_row + d, gmem_row + d, valid && (d + 8 <= D));
    }
}

// Load multiple rows with strided access pattern
template <int BLOCK_N, int D, int NUM_THREADS>
__device__ __forceinline__ void load_kv_block_async(
    __nv_bfloat16* smem_kv,  // (BLOCK_N, D) in shared memory
    const __nv_bfloat16* gmem_kv,  // Global memory pointer
    int gmem_stride,  // Stride between rows in global memory
    int num_valid_rows,  // Number of valid rows to load
    int tid  // Thread ID
) {
    // Each thread handles multiple elements
    constexpr int ELEMENTS_PER_THREAD = (BLOCK_N * D + NUM_THREADS - 1) / NUM_THREADS;

    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
        int idx = tid + i * NUM_THREADS;
        if (idx < BLOCK_N * D) {
            int row = idx / D;
            int col = idx % D;
            bool valid = row < num_valid_rows;

            if (valid) {
                smem_kv[row * D + col] = gmem_kv[row * gmem_stride + col];
            } else {
                smem_kv[row * D + col] = __float2bfloat16(0.0f);
            }
        }
    }
}

}  // namespace fused_look_around
