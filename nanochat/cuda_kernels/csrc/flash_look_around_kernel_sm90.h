// flash_look_around_kernel_sm90.h - Kernel entry point for FA3-style look-around attention
// Implements warp specialization pattern from FlashAttention-3 for Hopper (sm_90) and Blackwell (sm_120)
//
// Architecture:
// - Producer warp group (WG0): Loads Q, K (with halo), V via TMA
// - Consumer warp groups (WG1+): Compute GMMA operations (QK^T, P@V)
// - Overlapped execution hides memory latency

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/pipeline.hpp>

// WGMMA support for sm_90+
#include <cute/arch/mma_sm90_gmma.hpp>
#include <cute/arch/mma_sm90_desc.hpp>

#include "flash_look_around_softmax.h"

namespace flash_look_around {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Configuration constants
////////////////////////////////////////////////////////////////////////////////////////////////////

// Tile dimensions for attention computation
struct TileConfig {
    static constexpr int kBlockM = 64;      // Query tile size
    static constexpr int kBlockN = 64;      // Key/Value tile size
    static constexpr int kHeadDim = 64;     // Head dimension (also supports 128)
    static constexpr int kHalo = 4;         // Halo size (±2 for 5-tap convolution)
    static constexpr int kStages = 2;       // Pipeline depth for double buffering
    static constexpr int kConvTaps = 5;     // Number of convolution taps
};

// Thread configuration
struct ThreadConfig {
    static constexpr int kWarpSize = 32;
    static constexpr int kNumWarpsProducer = 1;   // Producer uses 1 warp for TMA
    static constexpr int kNumWarpsConsumer = 4;   // Consumer uses 4 warps for GMMA
    static constexpr int kNumWarps = kNumWarpsProducer + kNumWarpsConsumer;
    static constexpr int kNumThreads = kNumWarps * kWarpSize;  // 160 threads

    // For simpler implementation, use all warps for both loading and compute
    // This matches the non-WS (warp-specialized) FA3 pattern
    static constexpr int kNumThreadsSimple = 256;  // 8 warps for WMMA

    // WGMMA configuration: 1 warp group = 4 warps = 128 threads
    static constexpr int kNumThreadsWGMMA = 128;
    static constexpr int kNumWarpsWGMMA = 4;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// WGMMA Configuration
////////////////////////////////////////////////////////////////////////////////////////////////////

// WGMMA tile dimensions: 64x64x16 for optimal performance
struct WGMMAConfig {
    static constexpr int kTileM = 64;     // M dimension (query rows)
    static constexpr int kTileN = 64;     // N dimension (key/value cols)
    static constexpr int kTileK = 16;     // K dimension (reduction)

    // Number of accumulator registers per thread for 64x64 output tile
    // Each thread owns 32 floats = 64*64 / 128 threads
    static constexpr int kAccumRegs = 32;

    // Number of A register inputs (4 x uint32_t = 8 bf16 values per thread)
    static constexpr int kARegs = 4;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Shared memory storage
////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Element_, int kBlockM_, int kBlockN_, int kHeadDim_, int kHalo_, int kStages_>
struct TensorStorage {
    using Element = Element_;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kHalo = kHalo_;
    static constexpr int kStages = kStages_;

    // Halo padded to 16 for WMMA/WGMMA alignment
    static constexpr int kHaloPadded = 16;
    static constexpr int kBlockN_Halo = kBlockN + kHaloPadded;

    // Score matrix stride (with padded halo)
    static constexpr int kScoreStride = kBlockN_Halo;

    // Shared memory arrays (aligned to 128 bytes for optimal access)
    alignas(128) Element smem_q[kBlockM * kHeadDim];
    alignas(128) Element smem_k[kBlockN_Halo * kHeadDim * kStages];  // K with halo, multi-stage
    alignas(128) Element smem_v[kBlockN * kHeadDim * kStages];       // V, multi-stage
    alignas(128) float smem_s[kBlockM * kScoreStride];               // QK^T scores with halo
    alignas(128) Element smem_p[kBlockM * kBlockN];                  // Convolved P for P@V
    alignas(128) float smem_o_block[kBlockM * kHeadDim];             // Output block accumulator
    alignas(128) float smem_o_acc[kBlockM * kHeadDim];               // Running output accumulator

    // Per-row accumulators
    alignas(64) float smem_m[kBlockM];         // Running max
    alignas(64) float smem_l[kBlockM];         // Running sum
    alignas(64) float smem_m_block[kBlockM];   // Block max
    alignas(64) float smem_l_block[kBlockM];   // Block sum

    // Convolution weights (shared across all threads)
    alignas(16) float smem_conv_weights[5];
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Named barriers for synchronization
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class NamedBarriers : int {
    QueryLoaded = 0,
    ProducerWG = 1,
    ConsumerWG = 2,
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward kernel parameters
////////////////////////////////////////////////////////////////////////////////////////////////////

struct FlashLookAroundFwdParams {
    // Input tensors (bfloat16)
    void const* Q_ptr;
    void const* K_ptr;
    void const* V_ptr;
    void const* conv_weights_ptr;  // (n_kv_heads, 5) float32

    // Output tensors
    void* O_ptr;         // (B, n_q_heads, T_q, D) bfloat16
    float* LSE_ptr;      // (B, n_q_heads, T_q) float32

    // Dimensions
    int B;               // Batch size
    int n_q_heads;       // Number of query heads
    int n_kv_heads;      // Number of key/value heads (for GQA)
    int T_q;             // Query sequence length
    int T_k;             // Key/Value sequence length
    int D;               // Head dimension

    // Strides (in elements, not bytes)
    int64_t Q_batch_stride;
    int64_t Q_head_stride;
    int64_t Q_seq_stride;
    int64_t K_batch_stride;
    int64_t K_head_stride;
    int64_t K_seq_stride;
    int64_t V_batch_stride;
    int64_t V_head_stride;
    int64_t V_seq_stride;
    int64_t O_batch_stride;
    int64_t O_head_stride;
    int64_t O_seq_stride;

    // Attention parameters
    float sm_scale;      // 1/sqrt(D)
    bool causal;
    int window_left;     // -1 for full attention
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward kernel parameters
////////////////////////////////////////////////////////////////////////////////////////////////////

struct FlashLookAroundBwdParams {
    // Forward inputs
    void const* Q_ptr;
    void const* K_ptr;
    void const* V_ptr;
    void const* O_ptr;
    float const* LSE_ptr;
    void const* conv_weights_ptr;

    // Upstream gradient
    void const* dO_ptr;

    // Output gradients
    void* dQ_ptr;
    float* dK_ptr;       // float32 for atomic accumulation
    float* dV_ptr;       // float32 for atomic accumulation
    float* dConv_weights_ptr;

    // Dimensions (same as forward)
    int B, n_q_heads, n_kv_heads, T_q, T_k, D;

    // Strides
    int64_t Q_batch_stride, Q_head_stride, Q_seq_stride;
    int64_t K_batch_stride, K_head_stride, K_seq_stride;
    int64_t V_batch_stride, V_head_stride, V_seq_stride;
    int64_t O_batch_stride, O_head_stride, O_seq_stride;
    int64_t dO_batch_stride, dO_head_stride, dO_seq_stride;
    int64_t dQ_batch_stride, dQ_head_stride, dQ_seq_stride;
    int64_t dK_batch_stride, dK_head_stride, dK_seq_stride;
    int64_t dV_batch_stride, dV_head_stride, dV_seq_stride;

    float sm_scale;
    bool causal;
    int window_left;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert bfloat16 to float
__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

// Convert float to bfloat16
__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// WGMMA Helper Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert shared memory pointer to GMMA descriptor
// For WGMMA, B operand must be in shared memory with proper swizzling
// Layout type 1 = Swizzle128B (for 128-byte aligned data)
__device__ __forceinline__ uint64_t make_wgmma_desc(
    const void* smem_ptr,
    int leading_dim_bytes,    // Stride between rows in bytes
    int stride_dim_bytes = 0  // Stride between 8-row blocks (0 for contiguous)
) {
    cute::GmmaDescriptor desc;
    // Convert shared memory pointer to 32-bit address
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));

    // Start address: bits [4:17] of smem address (drop lowest 4 bits)
    desc.bitfield.start_address_ = smem_addr >> 4;

    // Leading dimension byte offset (bits [4:17])
    // For K-major (col-major for K), this is the stride between columns
    desc.bitfield.leading_byte_offset_ = leading_dim_bytes >> 4;

    // Stride dimension byte offset (bits [4:17])
    // For K-major, this is the stride between 8-row chunks
    desc.bitfield.stride_byte_offset_ = stride_dim_bytes >> 4;

    // Base offset for swizzle (0 for now)
    desc.bitfield.base_offset_ = 0;

    // Layout type: 1 = SWIZZLE_128B for optimal performance
    // Using 0 (INTERLEAVE) for non-swizzled layout initially
    desc.bitfield.layout_type_ = 0;

    return desc.desc_;
}

// Create descriptor for K matrix (col-major, i.e., K-major for GEMM)
// K is stored as (N+Halo, D) row-major, but we access it as K^T
// For Q @ K^T, K operand is transposed: read as (D, N+Halo) column-major
__device__ __forceinline__ uint64_t make_k_desc(
    const __nv_bfloat16* K_smem,
    int N_with_halo,  // Number of K positions (including halo)
    int D             // Head dimension
) {
    // K is stored row-major as (N_with_halo, D)
    // For K^T, leading dimension is D elements = D * 2 bytes
    int leading_bytes = D * sizeof(__nv_bfloat16);
    // Stride between 8-row blocks: 8 * D * 2 bytes
    int stride_bytes = 8 * D * sizeof(__nv_bfloat16);
    return make_wgmma_desc(K_smem, leading_bytes, stride_bytes);
}

// Create descriptor for V matrix (row-major for P @ V)
// V is stored as (N, D) row-major
// P @ V: P is (M, N), V is (N, D), output is (M, D)
__device__ __forceinline__ uint64_t make_v_desc(
    const __nv_bfloat16* V_smem,
    int N,  // Number of V positions
    int D   // Head dimension
) {
    // V stored row-major as (N, D)
    // For WGMMA, B operand (V) needs column-major within 8x8 tiles
    // Leading dimension is D elements = D * 2 bytes
    int leading_bytes = D * sizeof(__nv_bfloat16);
    int stride_bytes = 8 * D * sizeof(__nv_bfloat16);
    return make_wgmma_desc(V_smem, leading_bytes, stride_bytes);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward kernel declaration (implementation in flash_look_around_fwd_sm90.cuh)
////////////////////////////////////////////////////////////////////////////////////////////////////

// WMMA version (256 threads, 8 warps)
template <int kBlockM, int kBlockN, int kHeadDim, bool kIsCausal>
__global__ void flash_look_around_fwd_kernel_sm90(FlashLookAroundFwdParams params);

// WGMMA version (128 threads, 1 warp group)
template <int kBlockM, int kBlockN, int kHeadDim, bool kIsCausal>
__global__ void flash_look_around_fwd_kernel_wgmma(FlashLookAroundFwdParams params);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward kernel declaration (implementation in flash_look_around_bwd_sm90.cuh)
////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kBlockM, int kBlockN, int kHeadDim, bool kIsCausal>
__global__ void flash_look_around_bwd_kernel_sm90(FlashLookAroundBwdParams params);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Kernel launcher
////////////////////////////////////////////////////////////////////////////////////////////////////

// WMMA launcher (256 threads, 8 warps)
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
);

// WGMMA launcher (128 threads, 1 warp group) - requires sm_90+
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
);

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
);

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash_look_around
