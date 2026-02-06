// flash_look_around_kernel_sm120.h - Kernel entry point for SM120 (Blackwell) look-around attention
// Implements tcgen05 MMA with TMEM (Tensor Memory) for Blackwell GPUs
//
// Architecture differences from SM90 (Hopper):
// - tcgen05.mma instructions (single-thread issue model vs warp-group)
// - Accumulator in TMEM (256KB/SM) instead of registers
// - Supports 64x64 or 128x256 MMA tiles
// - Uses CUTLASS 4.x SM100_MMA_F16BF16_SS atoms
// - SM120 F8 MMA: SM120_16x8x32_TN for E4M3 x E4M3 with K=32 (2x throughput vs BF16 K=16)

#pragma once

#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

// SM90 header for shared types and params
#include "flash_look_around_kernel_sm90.h"

// SM100/SM120 UMMA support (tcgen05)
#if defined(ENABLE_SM120) && ENABLE_SM120
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/mma_sm100_desc.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#endif

// SM120 F8 MMA support (requires CUDA 12.8+ and __CUDA_ARCH__ >= 1200)
#if defined(ENABLE_SM120) && ENABLE_SM120
#include <cute/numeric/numeric_types.hpp>  // cute::float_e4m3_t
#include <cute/arch/mma_sm120.hpp>         // SM120_16x8x32_TN
#endif

// Note: We use inline PTX for cp.async and mbarrier instead of <cuda/pipeline>
// This provides better control and compatibility across CUDA versions

namespace flash_look_around {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// SM120 F8 Quantization and MMA Helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200 && defined(CUTE_ARCH_F8F6F4_MMA_ENABLED)

// Convert BF16 to E4M3 (FP8)
// E4M3 range: [-448, 448], with special handling for subnormals
__device__ __forceinline__ uint8_t bf16_to_e4m3(__nv_bfloat16 val) {
    float f = __bfloat162float(val);
    // Clamp to E4M3 representable range
    f = fmaxf(-448.0f, fminf(448.0f, f));
    return static_cast<uint8_t>(cutlass::float_e4m3_t(f).storage);
}

// Vectorized: convert 4 BF16 to 4 E4M3 packed in uint32
__device__ __forceinline__ uint32_t bf16x4_to_e4m3x4(
    __nv_bfloat16 v0, __nv_bfloat16 v1, __nv_bfloat16 v2, __nv_bfloat16 v3
) {
    return static_cast<uint32_t>(bf16_to_e4m3(v0)) |
           (static_cast<uint32_t>(bf16_to_e4m3(v1)) << 8) |
           (static_cast<uint32_t>(bf16_to_e4m3(v2)) << 16) |
           (static_cast<uint32_t>(bf16_to_e4m3(v3)) << 24);
}

// Convert float to E4M3 with pre-applied scale (for attention scores after softmax)
__device__ __forceinline__ uint8_t float_to_e4m3(float val) {
    val = fmaxf(-448.0f, fminf(448.0f, val));
    return static_cast<uint8_t>(cutlass::float_e4m3_t(val).storage);
}

// Vectorized: convert 4 floats to 4 E4M3 packed in uint32
__device__ __forceinline__ uint32_t floatx4_to_e4m3x4(float v0, float v1, float v2, float v3) {
    return static_cast<uint32_t>(float_to_e4m3(v0)) |
           (static_cast<uint32_t>(float_to_e4m3(v1)) << 8) |
           (static_cast<uint32_t>(float_to_e4m3(v2)) << 16) |
           (static_cast<uint32_t>(float_to_e4m3(v3)) << 24);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SM120 F8 MMA Configuration for 16x8x32 tiles
////////////////////////////////////////////////////////////////////////////////////////////////////

// SM120_16x8x32_TN register layout (E4M3 x E4M3 -> F32):
// - 4 A registers (each uint32 holds 4 E4M3 values = 16 values total)
// - 2 B registers (each uint32 holds 4 E4M3 values = 8 values total)
// - 4 D registers (float accumulators)
//
// Thread mapping in warp (32 threads):
// - A matrix (16 rows x 32 cols): threads cooperatively load fragments
// - B matrix (32 rows x 8 cols): threads cooperatively load fragments
// - D matrix (16 rows x 8 cols): 4 elements per thread

struct SM120_F8_MMA_Config {
    static constexpr int kMmaM = 16;   // M dimension of single MMA tile
    static constexpr int kMmaN = 8;    // N dimension of single MMA tile
    static constexpr int kMmaK = 32;   // K dimension (2x BF16's K=16)

    // Registers per thread for one MMA tile
    static constexpr int kARegs = 4;   // uint32 A registers (4 x 4 E4M3 = 16 values)
    static constexpr int kBRegs = 2;   // uint32 B registers (2 x 4 E4M3 = 8 values)
    static constexpr int kDRegs = 4;   // float accumulator registers

    // For 64x64 attention block with D=64:
    // QK^T: (64, 64) x (64, 80) -> (64, 80)  [80 = 64 + halo_padded]
    //   M_tiles = 64/16 = 4, N_tiles = 80/8 = 10, K_iters = 64/32 = 2
    // P@V:  (64, 64) x (64, 64) -> (64, 64)
    //   M_tiles = 4, N_tiles = 8, K_iters = 2
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// F8 MMA tile execution helper
// Computes one 16x8 output tile using SM120_16x8x32_TN
////////////////////////////////////////////////////////////////////////////////////////////////////

// Load A fragment (Q) for one 16x32 tile
// A is row-major, each thread loads 16 E4M3 values
// Thread mapping: threads 0-31 cooperatively load 16 rows x 32 cols
__device__ __forceinline__ void load_a_fragment_f8(
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3,
    const uint8_t* A_smem,  // Pointer to start of 16x32 tile in smem (E4M3)
    int A_stride,           // Stride in elements (not bytes)
    int lane_id
) {
    // SM120_16x8x32_TN register layout for A (16 rows x 32 cols):
    // Each thread owns 4 regs, each reg holds 4 consecutive E4M3 values along K
    // Thread lane_id owns row (lane_id / 4) for the first 8 rows, repeated
    // and column offset (lane_id % 4) * 8

    const int row_group = lane_id / 4;      // 0-7, which pair of rows
    const int col_group = lane_id % 4;      // 0-3, which K chunk (0-7, 8-15, 16-23, 24-31)

    // First set of rows (0-7)
    const uint8_t* row0_ptr = A_smem + row_group * A_stride + col_group * 8;
    // Second set of rows (8-15)
    const uint8_t* row1_ptr = A_smem + (row_group + 8) * A_stride + col_group * 8;

    // Load 4 E4M3 values at a time (vectorized)
    a0 = *reinterpret_cast<const uint32_t*>(row0_ptr);
    a1 = *reinterpret_cast<const uint32_t*>(row0_ptr + 4);
    a2 = *reinterpret_cast<const uint32_t*>(row1_ptr);
    a3 = *reinterpret_cast<const uint32_t*>(row1_ptr + 4);
}

// Load B fragment (K^T) for one 32x8 tile
// B is col-major (K stored row-major, read as K^T)
// Each thread loads 8 E4M3 values
__device__ __forceinline__ void load_b_fragment_f8(
    uint32_t& b0, uint32_t& b1,
    const uint8_t* B_smem,  // Pointer to start of 32x8 tile (stored as 8x32 row-major K)
    int B_stride,           // Stride in elements for K (row-major)
    int lane_id
) {
    // SM120_16x8x32_TN B matrix (32 rows x 8 cols, transposed from K):
    // K is stored (N, D) row-major, we need K^T which is (D, N)
    // For a 32x8 B tile, we read from K which is 8x32 in row-major

    // Thread lane_id owns: rows (lane_id * 1) to (lane_id * 1 + 0) for first 32 rows
    // and column (lane_id % 8)
    const int k_idx = lane_id;  // Which K dimension (0-31)
    const int n_offset = 0;     // Starting N offset (each thread handles 8 N values)

    // K is (N, D) row-major: K[n, k] = K_smem[n * D + k]
    // We need K^T[k, n] = K[n, k], so read column k across 8 rows
    const uint8_t* k_col = B_smem + k_idx;

    // Load 4 values from consecutive N positions (N=0-3, then N=4-7)
    uint8_t v0 = k_col[0 * B_stride];
    uint8_t v1 = k_col[1 * B_stride];
    uint8_t v2 = k_col[2 * B_stride];
    uint8_t v3 = k_col[3 * B_stride];
    b0 = static_cast<uint32_t>(v0) | (static_cast<uint32_t>(v1) << 8) |
         (static_cast<uint32_t>(v2) << 16) | (static_cast<uint32_t>(v3) << 24);

    uint8_t v4 = k_col[4 * B_stride];
    uint8_t v5 = k_col[5 * B_stride];
    uint8_t v6 = k_col[6 * B_stride];
    uint8_t v7 = k_col[7 * B_stride];
    b1 = static_cast<uint32_t>(v4) | (static_cast<uint32_t>(v5) << 8) |
         (static_cast<uint32_t>(v6) << 16) | (static_cast<uint32_t>(v7) << 24);
}

// Load B fragment for P@V where V is row-major and used directly (no transpose)
// V is (K, D) row-major: V[k, d] = V_smem[k * V_stride + d]
// MMA expects B as col-major (K=32, N=8), we read consecutive D values per K row
__device__ __forceinline__ void load_b_fragment_f8_direct(
    uint32_t& b0, uint32_t& b1,
    const uint8_t* V_tile,  // Pointer to start of 32x8 tile in V
    int V_stride,           // Stride between rows (BLOCK_D)
    int lane_id
) {
    // Thread lane_id handles k_idx = lane_id (0-31)
    // Read 8 consecutive D values from row k_idx
    const int k_idx = lane_id;
    const uint8_t* v_row = V_tile + k_idx * V_stride;

    // Load 8 bytes from row k_idx, columns 0-7
    uint8_t v0 = v_row[0];
    uint8_t v1 = v_row[1];
    uint8_t v2 = v_row[2];
    uint8_t v3 = v_row[3];
    uint8_t v4 = v_row[4];
    uint8_t v5 = v_row[5];
    uint8_t v6 = v_row[6];
    uint8_t v7 = v_row[7];

    // Pack into registers
    b0 = static_cast<uint32_t>(v0) | (static_cast<uint32_t>(v1) << 8) |
         (static_cast<uint32_t>(v2) << 16) | (static_cast<uint32_t>(v3) << 24);
    b1 = static_cast<uint32_t>(v4) | (static_cast<uint32_t>(v5) << 8) |
         (static_cast<uint32_t>(v6) << 16) | (static_cast<uint32_t>(v7) << 24);
}

// Store D fragment (output) from 4 float registers to smem
// Output is 16 rows x 8 cols, thread owns 4 elements
__device__ __forceinline__ void store_d_fragment_f8(
    float d0, float d1, float d2, float d3,
    float* D_smem,  // Pointer to start of 16x8 tile
    int D_stride,   // Stride in floats
    int lane_id
) {
    // Each thread owns 4 output elements
    // Thread layout: row = lane_id / 4 (first 8 rows), col = (lane_id % 4) * 2
    // and row + 8 for second set
    const int row_group = lane_id / 4;
    const int col_base = (lane_id % 4) * 2;

    // First pair of outputs (rows 0-7)
    D_smem[row_group * D_stride + col_base] = d0;
    D_smem[row_group * D_stride + col_base + 1] = d1;
    // Second pair of outputs (rows 8-15)
    D_smem[(row_group + 8) * D_stride + col_base] = d2;
    D_smem[(row_group + 8) * D_stride + col_base + 1] = d3;
}

// Execute one F8 MMA 16x8x32 operation
__device__ __forceinline__ void mma_f8_16x8x32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3
) {
    cute::SM120_16x8x32_TN<cute::float_e4m3_t, cute::float_e4m3_t, float>::fma(
        d0, d1, d2, d3,
        a0, a1, a2, a3,
        b0, b1,
        c0, c1, c2, c3
    );
}

#endif  // __CUDA_ARCH__ >= 1200 && CUTE_ARCH_F8F6F4_MMA_ENABLED

////////////////////////////////////////////////////////////////////////////////////////////////////
// SM120 Configuration
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM120Config {
    // MMA tile dimensions for tcgen05: 64x64x16 for F16/BF16
    static constexpr int kMmaM = 64;      // M dimension
    static constexpr int kMmaN = 64;      // N dimension
    static constexpr int kMmaK = 16;      // K dimension for F16/BF16

    // Block dimensions
    static constexpr int kBlockM = 64;
    static constexpr int kBlockN = 64;
    static constexpr int kHeadDim = 64;
    static constexpr int kHalo = 4;           // Halo for 5-tap convolution
    static constexpr int kHaloPadded = 16;    // Aligned for MMA

    // Thread configuration
    // tcgen05 uses single-thread issue but needs synchronization
    static constexpr int kNumThreads = 128;   // Use same thread count as WGMMA for loading
    static constexpr int kWarpSize = 32;
    static constexpr int kNumWarps = kNumThreads / kWarpSize;

    // Pipeline stages for TMA/async memory operations
    static constexpr int kStages = 3;         // 3-stage pipelining for memory latency hiding

    // TMEM allocation (managed by CUTLASS atoms)
    // Each SM120 SM has 256KB TMEM
    static constexpr int kTmemAccumSize = kMmaM * kMmaN * sizeof(float);
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// SM120 3-Stage Pipeline State
// Used to track which stage is being loaded vs computed
////////////////////////////////////////////////////////////////////////////////////////////////////

struct SM120PipelineState {
    int read_stage;     // Stage currently being computed
    int write_stage;    // Stage currently being loaded
    int k_block;        // Current K block index being processed

    __device__ __forceinline__ SM120PipelineState()
        : read_stage(0), write_stage(0), k_block(0) {}

    __device__ __forceinline__ void advance_read() {
        read_stage = (read_stage + 1) % SM120Config::kStages;
    }

    __device__ __forceinline__ void advance_write() {
        write_stage = (write_stage + 1) % SM120Config::kStages;
        k_block++;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// SM120 Shared Memory Storage with 3-Stage Pipeline Buffers
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element_, int kBlockM_, int kBlockN_, int kHeadDim_, int kHalo_, int kStages_>
struct SM120TensorStorage {
    using Element = Element_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHalo = kHalo_;
    static constexpr int kStages = kStages_;

    // Halo padded to 16 for MMA alignment
    static constexpr int kHaloPadded = 16;
    static constexpr int kBlockN_Halo = kBlockN + kHaloPadded;

    // Score matrix stride
    static constexpr int kScoreStride = kBlockN_Halo;

    // Size constants for each buffer
    static constexpr int kQSize = kBlockM * kHeadDim;
    static constexpr int kKStageSize = kBlockN_Halo * kHeadDim;  // Per-stage K size (with halo)
    static constexpr int kVStageSize = kBlockN * kHeadDim;       // Per-stage V size

    // ============================================================
    // Shared memory arrays (aligned to 128 bytes for optimal access)
    // ============================================================

    // Q is loaded once at the start (not pipelined)
    alignas(128) Element smem_q[kQSize];

    // K with halo - TRIPLE BUFFERED for 3-stage pipeline
    // Layout: [stage0][stage1][stage2], each stage has (kBlockN_Halo * kHeadDim) elements
    alignas(128) Element smem_k[kKStageSize * kStages];

    // V - TRIPLE BUFFERED for 3-stage pipeline
    // Layout: [stage0][stage1][stage2], each stage has (kBlockN * kHeadDim) elements
    alignas(128) Element smem_v[kVStageSize * kStages];

    // QK^T scores with halo (single buffer, recomputed each iteration)
    alignas(128) float smem_s[kBlockM * kScoreStride];

    // Convolved P for P@V (single buffer)
    alignas(128) Element smem_p[kBlockM * kBlockN];

    // Output block accumulator (single buffer)
    alignas(128) float smem_o_block[kBlockM * kHeadDim];

    // Running output accumulator
    alignas(128) float smem_o_acc[kBlockM * kHeadDim];

    // Per-row accumulators
    alignas(64) float smem_m[kBlockM];         // Running max
    alignas(64) float smem_l[kBlockM];         // Running sum
    alignas(64) float smem_m_block[kBlockM];   // Block max
    alignas(64) float smem_l_block[kBlockM];   // Block sum

    // Convolution weights
    alignas(16) float smem_conv_weights[5];

    // ============================================================
    // Pipeline barriers for async memory operations
    // One barrier per stage to track completion
    // ============================================================
    alignas(8) uint64_t pipe_barrier[kStages];

    // ============================================================
    // Helper functions to get pointers into staged buffers
    // ============================================================

    __device__ __forceinline__ Element* k_stage(int stage) {
        return smem_k + stage * kKStageSize;
    }

    __device__ __forceinline__ const Element* k_stage(int stage) const {
        return smem_k + stage * kKStageSize;
    }

    __device__ __forceinline__ Element* v_stage(int stage) {
        return smem_v + stage * kVStageSize;
    }

    __device__ __forceinline__ const Element* v_stage(int stage) const {
        return smem_v + stage * kVStageSize;
    }

    // ============================================================
    // Calculate total shared memory usage
    // ============================================================
    static constexpr size_t total_smem_bytes() {
        return sizeof(Element) * kQSize +                      // Q
               sizeof(Element) * kKStageSize * kStages +       // K (triple-buffered)
               sizeof(Element) * kVStageSize * kStages +       // V (triple-buffered)
               sizeof(float) * kBlockM * kScoreStride +        // S
               sizeof(Element) * kBlockM * kBlockN +           // P
               sizeof(float) * kBlockM * kHeadDim +            // O_block
               sizeof(float) * kBlockM * kHeadDim +            // O_acc
               sizeof(float) * kBlockM * 4 +                   // m, l, m_block, l_block
               sizeof(float) * 5 +                             // conv_weights
               sizeof(uint64_t) * kStages;                     // pipe_barrier
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// SM120 Async Pipeline Helpers
// Uses cp.async for asynchronous memory copies with mbarrier synchronization
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200

// Initialize pipeline barriers for 3-stage operation
__device__ __forceinline__ void init_pipeline_barriers(uint64_t* barriers, int num_stages, int expected_count) {
    // Only thread 0 initializes barriers
    if (threadIdx.x == 0) {
        for (int s = 0; s < num_stages; s++) {
            // Initialize as mbarrier with expected arrival count
            asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                : : "l"(&barriers[s]), "r"(expected_count));
        }
    }
    __syncthreads();
}

// Arrive at barrier (producer signals completion of async copy)
__device__ __forceinline__ void arrive_barrier(uint64_t* barrier, int count = 1) {
    asm volatile("mbarrier.arrive.shared.b64 _, [%0], %1;"
        : : "l"(barrier), "r"(count));
}

// Wait on barrier (consumer waits for data to be ready)
__device__ __forceinline__ void wait_barrier(uint64_t* barrier, int phase) {
    // Wait for barrier to reach expected phase
    uint32_t mbar_wait;
    asm volatile(
        "{\n"
        ".reg .pred P1;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared.b64 P1, [%1], %2;\n"
        "@!P1 bra LAB_WAIT;\n"
        "}"
        : "=r"(mbar_wait) : "l"(barrier), "r"(phase));
}

// Async copy with cp.async - copies from global to shared memory
// Uses vectorized 16-byte copies for optimal bandwidth
template<typename T>
__device__ __forceinline__ void cp_async_tile(
    T* smem_ptr,
    const T* gmem_ptr,
    int num_elements,
    int tid,
    int num_threads,
    bool predicate = true
) {
    // Each thread handles a portion of elements
    // Use cp.async.cg for coalesced global memory access (16 bytes = 8 bf16)
    constexpr int kVecSize = 8;  // 8 bf16 = 16 bytes per cp.async

    const int elements_per_thread = (num_elements + num_threads - 1) / num_threads;
    const int start_elem = tid * elements_per_thread;
    const int end_elem = min(start_elem + elements_per_thread, num_elements);

    for (int i = start_elem; i < end_elem; i += kVecSize) {
        if (predicate && (i + kVecSize <= num_elements)) {
            // 16-byte async copy
            asm volatile(
                "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;"
                : : "l"(smem_ptr + i), "l"(gmem_ptr + i));
        } else if (predicate) {
            // Handle remaining elements with scalar copies
            for (int j = i; j < end_elem && j < num_elements; j++) {
                smem_ptr[j] = gmem_ptr[j];
            }
        }
    }
}

// Commit all pending cp.async operations for this thread
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;");
}

// Wait for cp.async operations to complete
// N=0 means wait for all, N>0 means allow N groups to be outstanding
// Template version for compile-time known N
template<int N>
__device__ __forceinline__ void cp_async_wait_n() {
    if constexpr (N == 0) {
        asm volatile("cp.async.wait_all;");
    } else {
        asm volatile("cp.async.wait_group %0;" : : "n"(N));
    }
}

// Convenience alias for wait all
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;");
}

// Combined async load of K and V for one pipeline stage
template<int kBlockN, int kBlockD, int kHaloPadded>
__device__ __forceinline__ void load_kv_async(
    __nv_bfloat16* K_smem_stage,
    __nv_bfloat16* V_smem_stage,
    const __nv_bfloat16* K_batch,
    const __nv_bfloat16* V_batch,
    int k_block_start,
    int T_k,
    int D,
    int tid,
    int num_threads
) {
    const int K_size = (kBlockN + kHaloPadded) * kBlockD;
    const int V_size = kBlockN * kBlockD;

    // Load K with halo using cp.async
    // K layout: (N_with_halo, D) row-major
    for (int i = tid; i < K_size; i += num_threads) {
        int n = i / kBlockD;
        int d = i % kBlockD;
        int k_pos = k_block_start - kHaloPadded / 2 + n;

        if (k_pos >= 0 && k_pos < T_k && d < D) {
            // Valid position - use async copy
            const __nv_bfloat16* src = K_batch + k_pos * D + d;
            __nv_bfloat16* dst = K_smem_stage + i;
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 2;"
                : : "l"(dst), "l"(src));
        } else {
            // Out of bounds - zero pad (must be sync)
            K_smem_stage[i] = __float2bfloat16(0.0f);
        }
    }

    // Load V using cp.async
    // V layout: (N, D) row-major
    for (int i = tid; i < V_size; i += num_threads) {
        int n = i / kBlockD;
        int d = i % kBlockD;
        int v_pos = k_block_start + n;

        if (v_pos < T_k && d < D) {
            const __nv_bfloat16* src = V_batch + v_pos * D + d;
            __nv_bfloat16* dst = V_smem_stage + i;
            asm volatile(
                "cp.async.ca.shared.global [%0], [%1], 2;"
                : : "l"(dst), "l"(src));
        } else {
            V_smem_stage[i] = __float2bfloat16(0.0f);
        }
    }

    // Commit the async group
    cp_async_commit();
}

#endif  // __CUDA_ARCH__ >= 1200

////////////////////////////////////////////////////////////////////////////////////////////////////
// SM120 UMMA Descriptor Helpers
////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(ENABLE_SM120) && ENABLE_SM120

// Create UMMA descriptor for SMEM operand
__device__ __forceinline__ uint64_t make_umma_desc_sm120(
    const void* smem_ptr,
    int leading_dim_bytes,
    int stride_dim_bytes,
    UMMA::LayoutType layout_type = UMMA::LayoutType::SWIZZLE_NONE
) {
    UMMA::SmemDescriptor desc;
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    // Start address: bits [0,14), 4 LSB dropped
    desc.start_address_ = smem_addr >> 4;

    // Leading dimension byte offset
    desc.leading_byte_offset_ = leading_dim_bytes >> 4;

    // Stride dimension byte offset
    desc.stride_byte_offset_ = stride_dim_bytes >> 4;

    // Layout type
    desc.layout_type_ = static_cast<uint8_t>(layout_type);

    return desc.desc_;
}

// Create K matrix descriptor for Q @ K^T
// K stored as (N_with_halo, D) row-major, read as K^T
__device__ __forceinline__ uint64_t make_k_desc_sm120(
    const __nv_bfloat16* K_smem,
    int N_with_halo,
    int D
) {
    int leading_bytes = D * sizeof(__nv_bfloat16);
    int stride_bytes = 8 * D * sizeof(__nv_bfloat16);  // 8-row stride
    return make_umma_desc_sm120(K_smem, leading_bytes, stride_bytes);
}

// Create V matrix descriptor for P @ V
// V stored as (N, D) row-major
__device__ __forceinline__ uint64_t make_v_desc_sm120(
    const __nv_bfloat16* V_smem,
    int N,
    int D
) {
    int leading_bytes = D * sizeof(__nv_bfloat16);
    int stride_bytes = 8 * D * sizeof(__nv_bfloat16);
    return make_umma_desc_sm120(V_smem, leading_bytes, stride_bytes);
}

#endif  // ENABLE_SM120

////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward kernel declaration (implementation in flash_look_around_fwd_sm120.cuh)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN, int kHeadDim, bool kIsCausal>
__global__ void flash_look_around_fwd_kernel_sm120(FlashLookAroundFwdParams params);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel launcher for SM120
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
);

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash_look_around
