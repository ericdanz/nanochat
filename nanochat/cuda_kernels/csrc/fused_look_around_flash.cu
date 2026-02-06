// fused_look_around_flash.cu - Main CUDA file for fused look-around flash attention
// Provides entry points for forward and backward passes
//
// Includes two implementations:
// 1. WMMA-based kernel (for sm_70+) - current default
// 2. FA3-style kernel (for sm_90+) - uses better architecture patterns

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cmath>

// Original WMMA-based kernels
#include "fused_look_around_fwd.cuh"
#include "fused_look_around_bwd.cuh"

// New FA3-style kernels (sm_90+)
#include "flash_look_around_fwd_sm90.cuh"
#include "flash_look_around_bwd_sm90.cuh"

// SM120 (Blackwell) kernels with tcgen05/TMEM
#if defined(ENABLE_SM120) && ENABLE_SM120
#include "flash_look_around_fwd_sm120.cuh"
#include "flash_look_around_fwd_sm120_pipelined.cuh"
#include "flash_look_around_fwd_sm120_cute.cuh"
#endif

// Check CUDA errors
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward pass entry point
// Supports GQA: q has n_q_heads, k/v have n_kv_heads where n_q_heads % n_kv_heads == 0
std::vector<torch::Tensor> fused_look_around_flash_fwd_cuda(
    torch::Tensor q,           // (B, n_q_heads, T_q, D) bfloat16
    torch::Tensor k,           // (B, n_kv_heads, T_k, D) bfloat16
    torch::Tensor v,           // (B, n_kv_heads, T_k, D) bfloat16
    torch::Tensor proj_weights, // (n_kv_heads, 5) float32, pre-softmaxed
    bool causal,
    int window_left            // -1 for full attention, >= 0 for sliding window
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(proj_weights);

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(proj_weights.dtype() == torch::kFloat32, "proj_weights must be float32");

    const int B = q.size(0);
    const int n_q_heads = q.size(1);
    const int T_q = q.size(2);
    const int D = q.size(3);
    const int n_kv_heads = k.size(1);
    const int T_k = k.size(2);

    // GQA validation: n_q_heads must be divisible by n_kv_heads
    TORCH_CHECK(n_q_heads % n_kv_heads == 0,
                "n_q_heads (", n_q_heads, ") must be divisible by n_kv_heads (", n_kv_heads, ")");

    TORCH_CHECK(k.size(0) == B && k.size(3) == D,
                "k shape mismatch");
    TORCH_CHECK(v.size(0) == B && v.size(1) == n_kv_heads && v.size(2) == T_k && v.size(3) == D,
                "v shape mismatch");
    TORCH_CHECK(proj_weights.size(0) == n_kv_heads && proj_weights.size(1) == 5,
                "proj_weights must be (n_kv_heads, 5)");

    // Allocate output tensors
    auto out = torch::empty_like(q);
    auto lse = torch::empty({B, n_q_heads, T_q}, q.options().dtype(torch::kFloat32));

    float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));

    // Get CUDA stream
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    fused_look_around::launch_fused_look_around_flash_fwd(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        proj_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        B, n_q_heads, n_kv_heads, T_q, T_k, D,
        sm_scale,
        causal,
        window_left,
        stream
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch error: ", cudaGetErrorString(err));

    // Synchronize to catch execution errors
    err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel execution error: ", cudaGetErrorString(err));

    return {out, lse};
}

// Backward pass entry point
// Supports GQA: q has n_q_heads, k/v have n_kv_heads where n_q_heads % n_kv_heads == 0
std::vector<torch::Tensor> fused_look_around_flash_bwd_cuda(
    torch::Tensor grad_out,    // (B, n_q_heads, T_q, D) bfloat16
    torch::Tensor q,           // (B, n_q_heads, T_q, D) bfloat16
    torch::Tensor k,           // (B, n_kv_heads, T_k, D) bfloat16
    torch::Tensor v,           // (B, n_kv_heads, T_k, D) bfloat16
    torch::Tensor out,         // (B, n_q_heads, T_q, D) bfloat16
    torch::Tensor lse,         // (B, n_q_heads, T_q) float32
    torch::Tensor proj_weights, // (n_kv_heads, 5) float32, pre-softmaxed
    bool causal,
    int window_left            // -1 for full attention, >= 0 for sliding window
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(out);
    CHECK_INPUT(lse);
    CHECK_INPUT(proj_weights);

    const int B = q.size(0);
    const int n_q_heads = q.size(1);
    const int T_q = q.size(2);
    const int D = q.size(3);
    const int n_kv_heads = k.size(1);
    const int T_k = k.size(2);

    // GQA validation
    TORCH_CHECK(n_q_heads % n_kv_heads == 0,
                "n_q_heads (", n_q_heads, ") must be divisible by n_kv_heads (", n_kv_heads, ")");

    // Allocate gradient tensors
    auto grad_q = torch::empty_like(q);
    // Use float32 for K and V gradients for atomic adds, then convert
    auto grad_k_float = torch::zeros({B, n_kv_heads, T_k, D}, q.options().dtype(torch::kFloat32));
    auto grad_v_float = torch::zeros({B, n_kv_heads, T_k, D}, q.options().dtype(torch::kFloat32));
    auto grad_proj = torch::zeros({n_kv_heads, 5}, q.options().dtype(torch::kFloat32));

    float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch kernel
    fused_look_around::launch_fused_look_around_flash_bwd(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        proj_weights.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(grad_q.data_ptr<at::BFloat16>()),
        grad_k_float.data_ptr<float>(),
        grad_v_float.data_ptr<float>(),
        grad_proj.data_ptr<float>(),
        B, n_q_heads, n_kv_heads, T_q, T_k, D,
        sm_scale,
        causal,
        window_left,
        stream
    );

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA backward kernel launch error: ", cudaGetErrorString(err));

    // Synchronize to catch execution errors
    err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "CUDA backward kernel execution error: ", cudaGetErrorString(err));

    // Convert gradients back to bfloat16
    auto grad_k = grad_k_float.to(torch::kBFloat16);
    auto grad_v = grad_v_float.to(torch::kBFloat16);

    return {grad_q, grad_k, grad_v, grad_proj};
}

// ============================================================
// FA3-STYLE KERNEL ENTRY POINTS (sm_90+)
// These provide the new FlashAttention-3 style implementation
// ============================================================

// FA3-style forward pass entry point
std::vector<torch::Tensor> flash_look_around_fwd_sm90_cuda(
    torch::Tensor q,           // (B, n_q_heads, T_q, D) bfloat16
    torch::Tensor k,           // (B, n_kv_heads, T_k, D) bfloat16
    torch::Tensor v,           // (B, n_kv_heads, T_k, D) bfloat16
    torch::Tensor proj_weights, // (n_kv_heads, 5) float32, pre-softmaxed
    bool causal,
    int window_left
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(proj_weights);

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(proj_weights.dtype() == torch::kFloat32, "proj_weights must be float32");

    const int B = q.size(0);
    const int n_q_heads = q.size(1);
    const int T_q = q.size(2);
    const int D = q.size(3);
    const int n_kv_heads = k.size(1);
    const int T_k = k.size(2);

    TORCH_CHECK(n_q_heads % n_kv_heads == 0,
                "n_q_heads (", n_q_heads, ") must be divisible by n_kv_heads (", n_kv_heads, ")");
    TORCH_CHECK(D == 64 || D == 128, "Head dimension must be 64 or 128 for FA3 kernel, got ", D);

    // Allocate output tensors
    auto out = torch::empty_like(q);
    auto lse = torch::empty({B, n_q_heads, T_q}, q.options().dtype(torch::kFloat32));

    float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch FA3-style kernel
    flash_look_around::launch_flash_look_around_fwd_sm90(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        proj_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        B, n_q_heads, n_kv_heads, T_q, T_k, D,
        sm_scale,
        causal,
        window_left,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "FA3 CUDA kernel launch error: ", cudaGetErrorString(err));

    err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "FA3 CUDA kernel execution error: ", cudaGetErrorString(err));

    return {out, lse};
}

// FA3-style backward pass entry point
std::vector<torch::Tensor> flash_look_around_bwd_sm90_cuda(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor lse,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(out);
    CHECK_INPUT(lse);
    CHECK_INPUT(proj_weights);

    const int B = q.size(0);
    const int n_q_heads = q.size(1);
    const int T_q = q.size(2);
    const int D = q.size(3);
    const int n_kv_heads = k.size(1);
    const int T_k = k.size(2);

    TORCH_CHECK(n_q_heads % n_kv_heads == 0,
                "n_q_heads (", n_q_heads, ") must be divisible by n_kv_heads (", n_kv_heads, ")");
    TORCH_CHECK(D == 64 || D == 128, "Head dimension must be 64 or 128 for FA3 kernel, got ", D);

    // Allocate gradient tensors
    auto grad_q = torch::empty_like(q);
    auto grad_k_float = torch::zeros({B, n_kv_heads, T_k, D}, q.options().dtype(torch::kFloat32));
    auto grad_v_float = torch::zeros({B, n_kv_heads, T_k, D}, q.options().dtype(torch::kFloat32));
    auto grad_proj = torch::zeros({n_kv_heads, 5}, q.options().dtype(torch::kFloat32));

    float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch FA3-style backward kernel
    flash_look_around::launch_flash_look_around_bwd_sm90(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        proj_weights.data_ptr<float>(),
        reinterpret_cast<const __nv_bfloat16*>(grad_out.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(grad_q.data_ptr<at::BFloat16>()),
        grad_k_float.data_ptr<float>(),
        grad_v_float.data_ptr<float>(),
        grad_proj.data_ptr<float>(),
        B, n_q_heads, n_kv_heads, T_q, T_k, D,
        sm_scale,
        causal,
        window_left,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "FA3 CUDA backward kernel launch error: ", cudaGetErrorString(err));

    err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "FA3 CUDA backward kernel execution error: ", cudaGetErrorString(err));

    auto grad_k = grad_k_float.to(torch::kBFloat16);
    auto grad_v = grad_v_float.to(torch::kBFloat16);

    return {grad_q, grad_k, grad_v, grad_proj};
}

// Helper to check if FA3 kernel is available (sm_90+)
bool is_fa3_kernel_available() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.major >= 9;  // sm_90+ (Hopper or newer)
}

// ============================================================
// WGMMA KERNEL ENTRY POINT (sm_90a only - Hopper with async features)
// Uses warp group MMA for better tensor core utilization
// NOTE: Currently only works on Hopper (sm_90a). Blackwell (sm_120) has
// different GMMA instructions that are not yet supported in this kernel.
// ============================================================

std::vector<torch::Tensor> flash_look_around_fwd_wgmma_cuda(
    torch::Tensor q,           // (B, n_q_heads, T_q, D) bfloat16
    torch::Tensor k,           // (B, n_kv_heads, T_k, D) bfloat16
    torch::Tensor v,           // (B, n_kv_heads, T_k, D) bfloat16
    torch::Tensor proj_weights, // (n_kv_heads, 5) float32, pre-softmaxed
    bool causal,
    int window_left
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(proj_weights);

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(proj_weights.dtype() == torch::kFloat32, "proj_weights must be float32");

    const int B = q.size(0);
    const int n_q_heads = q.size(1);
    const int T_q = q.size(2);
    const int D = q.size(3);
    const int n_kv_heads = k.size(1);
    const int T_k = k.size(2);

    TORCH_CHECK(n_q_heads % n_kv_heads == 0,
                "n_q_heads (", n_q_heads, ") must be divisible by n_kv_heads (", n_kv_heads, ")");
    TORCH_CHECK(D == 64 || D == 128, "Head dimension must be 64 or 128 for WGMMA kernel, got ", D);

    // Check for sm_90a specifically (Hopper with async features)
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    // WGMMA requires sm_90a (Hopper with async features)
    // sm_120 (Blackwell) has different GMMA instructions not yet supported
    TORCH_CHECK(props.major == 9 && props.minor == 0,
                "WGMMA kernel requires sm_90a (Hopper). Got sm_", props.major, props.minor,
                ". For Blackwell (sm_120), use forward_sm90 instead which uses WMMA.");

    // Allocate output tensors
    auto out = torch::empty_like(q);
    auto lse = torch::empty({B, n_q_heads, T_q}, q.options().dtype(torch::kFloat32));

    float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch WGMMA kernel
    flash_look_around::launch_flash_look_around_fwd_wgmma(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        proj_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        B, n_q_heads, n_kv_heads, T_q, T_k, D,
        sm_scale,
        causal,
        window_left,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "WGMMA CUDA kernel launch error: ", cudaGetErrorString(err));

    err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "WGMMA CUDA kernel execution error: ", cudaGetErrorString(err));

    return {out, lse};
}

// ============================================================
// SM120 (BLACKWELL) KERNEL ENTRY POINT
// Uses tcgen05 MMA instructions with TMEM accumulator
// ============================================================

#if defined(ENABLE_SM120) && ENABLE_SM120

std::vector<torch::Tensor> flash_look_around_fwd_sm120_cuda(
    torch::Tensor q,           // (B, n_q_heads, T_q, D) bfloat16
    torch::Tensor k,           // (B, n_kv_heads, T_k, D) bfloat16
    torch::Tensor v,           // (B, n_kv_heads, T_k, D) bfloat16
    torch::Tensor proj_weights, // (n_kv_heads, 5) float32, pre-softmaxed
    bool causal,
    int window_left
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(proj_weights);

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(proj_weights.dtype() == torch::kFloat32, "proj_weights must be float32");

    const int B = q.size(0);
    const int n_q_heads = q.size(1);
    const int T_q = q.size(2);
    const int D = q.size(3);
    const int n_kv_heads = k.size(1);
    const int T_k = k.size(2);

    TORCH_CHECK(n_q_heads % n_kv_heads == 0,
                "n_q_heads (", n_q_heads, ") must be divisible by n_kv_heads (", n_kv_heads, ")");
    TORCH_CHECK(D == 64 || D == 128, "Head dimension must be 64 or 128 for SM120 kernel, got ", D);

    // Check for sm_120 (Blackwell)
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    TORCH_CHECK(props.major == 12,
                "SM120 kernel requires Blackwell GPU (sm_120). Got sm_", props.major, props.minor);

    // Allocate output tensors
    auto out = torch::empty_like(q);
    auto lse = torch::empty({B, n_q_heads, T_q}, q.options().dtype(torch::kFloat32));

    float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch SM120 kernel
    flash_look_around::launch_flash_look_around_fwd_sm120(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        proj_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        B, n_q_heads, n_kv_heads, T_q, T_k, D,
        sm_scale,
        causal,
        window_left,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "SM120 CUDA kernel launch error: ", cudaGetErrorString(err));

    err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "SM120 CUDA kernel execution error: ", cudaGetErrorString(err));

    return {out, lse};
}

// SM120 Pipelined kernel entry point (2-stage pipeline for better memory hiding)
std::vector<torch::Tensor> flash_look_around_fwd_sm120_pipelined_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(proj_weights);

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(proj_weights.dtype() == torch::kFloat32, "proj_weights must be float32");

    const int B = q.size(0);
    const int n_q_heads = q.size(1);
    const int T_q = q.size(2);
    const int D = q.size(3);
    const int n_kv_heads = k.size(1);
    const int T_k = k.size(2);

    TORCH_CHECK(n_q_heads % n_kv_heads == 0,
                "n_q_heads (", n_q_heads, ") must be divisible by n_kv_heads (", n_kv_heads, ")");
    TORCH_CHECK(D == 64, "Head dimension must be 64 for SM120 pipelined kernel, got ", D);

    // Check for sm_120 (Blackwell)
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    TORCH_CHECK(props.major == 12,
                "SM120 pipelined kernel requires Blackwell GPU (sm_120). Got sm_", props.major, props.minor);

    // Allocate output tensors
    auto out = torch::empty_like(q);
    auto lse = torch::empty({B, n_q_heads, T_q}, q.options().dtype(torch::kFloat32));

    float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch SM120 pipelined kernel
    flash_look_around::launch_flash_look_around_fwd_sm120_pipelined(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        proj_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        B, n_q_heads, n_kv_heads, T_q, T_k, D,
        sm_scale,
        causal,
        window_left,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "SM120 pipelined CUDA kernel launch error: ", cudaGetErrorString(err));

    err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "SM120 pipelined CUDA kernel execution error: ", cudaGetErrorString(err));

    return {out, lse};
}

// SM120 CuTe hybrid kernel entry point (CuTe TiledMMA + custom softmax/conv)
std::vector<torch::Tensor> flash_look_around_fwd_sm120_cute_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(proj_weights);

    TORCH_CHECK(q.dtype() == torch::kBFloat16, "q must be bfloat16");
    TORCH_CHECK(k.dtype() == torch::kBFloat16, "k must be bfloat16");
    TORCH_CHECK(v.dtype() == torch::kBFloat16, "v must be bfloat16");
    TORCH_CHECK(proj_weights.dtype() == torch::kFloat32, "proj_weights must be float32");

    const int B = q.size(0);
    const int n_q_heads = q.size(1);
    const int T_q = q.size(2);
    const int D = q.size(3);
    const int n_kv_heads = k.size(1);
    const int T_k = k.size(2);

    TORCH_CHECK(n_q_heads % n_kv_heads == 0,
                "n_q_heads (", n_q_heads, ") must be divisible by n_kv_heads (", n_kv_heads, ")");
    TORCH_CHECK(D == 64, "Head dimension must be 64 for SM120 CuTe kernel, got ", D);

    // Check for sm_120 (Blackwell)
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    TORCH_CHECK(props.major == 12,
                "SM120 CuTe kernel requires Blackwell GPU (sm_120). Got sm_", props.major, props.minor);

    // Allocate output tensors
    auto out = torch::empty_like(q);
    auto lse = torch::empty({B, n_q_heads, T_q}, q.options().dtype(torch::kFloat32));

    float sm_scale = 1.0f / std::sqrt(static_cast<float>(D));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Launch SM120 CuTe hybrid kernel
    flash_look_around::launch_flash_look_around_fwd_sm120_cute(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>()),
        proj_weights.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        lse.data_ptr<float>(),
        B, n_q_heads, n_kv_heads, T_q, T_k, D,
        sm_scale,
        causal,
        window_left,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "SM120 CuTe CUDA kernel launch error: ", cudaGetErrorString(err));

    err = cudaStreamSynchronize(stream);
    TORCH_CHECK(err == cudaSuccess, "SM120 CuTe CUDA kernel execution error: ", cudaGetErrorString(err));

    return {out, lse};
}

#endif  // ENABLE_SM120

// Helper to check if SM120 kernel is available
bool is_sm120_kernel_available() {
#if defined(ENABLE_SM120) && ENABLE_SM120
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);
    return props.major == 12;  // sm_120 (Blackwell)
#else
    return false;
#endif
}

// Auto-dispatch forward kernel based on GPU architecture
std::vector<torch::Tensor> flash_look_around_fwd_auto_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device);

    if (props.major == 12) {
        // Blackwell (RTX 5090): Use SM120 pipelined WMMA kernel
        // Note: SM120 F8 MMA kernel has V loading bugs, using pipelined WMMA instead
        return flash_look_around_fwd_sm120_pipelined_cuda(q, k, v, proj_weights, causal, window_left);
    }

    if (props.major == 9 && props.minor == 0) {
        // Hopper: Use WGMMA kernel
        return flash_look_around_fwd_wgmma_cuda(q, k, v, proj_weights, causal, window_left);
    }

    if (props.major >= 9) {
        // sm_90+ but not Hopper: Use WMMA-based FA3 kernel
        return flash_look_around_fwd_sm90_cuda(q, k, v, proj_weights, causal, window_left);
    }

    // Fallback to original WMMA kernel
    return fused_look_around_flash_fwd_cuda(q, k, v, proj_weights, causal, window_left);
}

// Module registration is in bindings.cpp
