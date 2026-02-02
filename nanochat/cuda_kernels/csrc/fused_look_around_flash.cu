// fused_look_around_flash.cu - Main CUDA file for fused look-around flash attention
// Provides entry points for forward and backward passes

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cmath>

#include "fused_look_around_fwd.cuh"
#include "fused_look_around_bwd.cuh"

// Check CUDA errors
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Forward pass entry point
std::vector<torch::Tensor> fused_look_around_flash_fwd_cuda(
    torch::Tensor q,           // (B, H, T_q, D) bfloat16
    torch::Tensor k,           // (B, H, T_k, D) bfloat16
    torch::Tensor v,           // (B, H, T_k, D) bfloat16
    torch::Tensor proj_weights, // (H, 5) float32, pre-softmaxed
    bool causal
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
    const int H = q.size(1);
    const int T_q = q.size(2);
    const int D = q.size(3);
    const int T_k = k.size(2);

    TORCH_CHECK(k.size(0) == B && k.size(1) == H && k.size(3) == D,
                "k shape mismatch");
    TORCH_CHECK(v.size(0) == B && v.size(1) == H && v.size(2) == T_k && v.size(3) == D,
                "v shape mismatch");
    TORCH_CHECK(proj_weights.size(0) == H && proj_weights.size(1) == 5,
                "proj_weights must be (H, 5)");

    // Allocate output tensors
    auto out = torch::empty_like(q);
    auto lse = torch::empty({B, H, T_q}, q.options().dtype(torch::kFloat32));

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
        B, H, T_q, T_k, D,
        sm_scale,
        causal,
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
std::vector<torch::Tensor> fused_look_around_flash_bwd_cuda(
    torch::Tensor grad_out,    // (B, H, T_q, D) bfloat16
    torch::Tensor q,           // (B, H, T_q, D) bfloat16
    torch::Tensor k,           // (B, H, T_k, D) bfloat16
    torch::Tensor v,           // (B, H, T_k, D) bfloat16
    torch::Tensor out,         // (B, H, T_q, D) bfloat16
    torch::Tensor lse,         // (B, H, T_q) float32
    torch::Tensor proj_weights, // (H, 5) float32, pre-softmaxed
    bool causal
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(out);
    CHECK_INPUT(lse);
    CHECK_INPUT(proj_weights);

    const int B = q.size(0);
    const int H = q.size(1);
    const int T_q = q.size(2);
    const int D = q.size(3);
    const int T_k = k.size(2);

    // Allocate gradient tensors
    auto grad_q = torch::empty_like(q);
    // Use float32 for K and V gradients for atomic adds, then convert
    auto grad_k_float = torch::zeros({B, H, T_k, D}, q.options().dtype(torch::kFloat32));
    auto grad_v_float = torch::zeros({B, H, T_k, D}, q.options().dtype(torch::kFloat32));
    auto grad_proj = torch::zeros({H, 5}, q.options().dtype(torch::kFloat32));

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
        B, H, T_q, T_k, D,
        sm_scale,
        causal,
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

// Module registration is in bindings.cpp
