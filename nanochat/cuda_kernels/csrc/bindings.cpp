// bindings.cpp - PyTorch bindings for fused look-around flash attention

#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions - Original WMMA kernel
std::vector<torch::Tensor> fused_look_around_flash_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
);

std::vector<torch::Tensor> fused_look_around_flash_bwd_cuda(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor lse,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
);

// Forward declarations of FA3-style CUDA functions (sm_90+)
std::vector<torch::Tensor> flash_look_around_fwd_sm90_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
);

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
);

// WGMMA forward (sm_90+, uses warp group MMA)
std::vector<torch::Tensor> flash_look_around_fwd_wgmma_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
);

// SM120 (Blackwell) forward - uses tcgen05 MMA with TMEM
#if defined(ENABLE_SM120) && ENABLE_SM120
std::vector<torch::Tensor> flash_look_around_fwd_sm120_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
);

// SM120 (Blackwell) pipelined forward - uses WMMA with cp.async 2-stage pipeline
std::vector<torch::Tensor> flash_look_around_fwd_sm120_pipelined_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
);

// SM120 (Blackwell) CuTe hybrid kernel - uses CuTe TiledMMA + custom softmax/conv
std::vector<torch::Tensor> flash_look_around_fwd_sm120_cute_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
);
#endif

// Auto-dispatch forward based on GPU architecture
std::vector<torch::Tensor> flash_look_around_fwd_auto_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor proj_weights,
    bool causal,
    int window_left
);

bool is_fa3_kernel_available();
bool is_sm120_kernel_available();

// Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Original WMMA kernel (default)
    m.def("forward", &fused_look_around_flash_fwd_cuda,
          "Fused look-around flash attention forward (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("proj_weights"), py::arg("causal") = true,
          py::arg("window_left") = -1);

    m.def("backward", &fused_look_around_flash_bwd_cuda,
          "Fused look-around flash attention backward (CUDA)",
          py::arg("grad_out"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("out"), py::arg("lse"), py::arg("proj_weights"),
          py::arg("causal") = true, py::arg("window_left") = -1);

    // FA3-style kernel (sm_90+)
    m.def("forward_sm90", &flash_look_around_fwd_sm90_cuda,
          "FA3-style look-around flash attention forward (CUDA, sm_90+)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("proj_weights"), py::arg("causal") = true,
          py::arg("window_left") = -1);

    m.def("backward_sm90", &flash_look_around_bwd_sm90_cuda,
          "FA3-style look-around flash attention backward (CUDA, sm_90+)",
          py::arg("grad_out"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("out"), py::arg("lse"), py::arg("proj_weights"),
          py::arg("causal") = true, py::arg("window_left") = -1);

    m.def("is_fa3_available", &is_fa3_kernel_available,
          "Check if FA3-style kernel is available (requires sm_90+)");

    // WGMMA kernel (sm_90+) - uses warp group MMA for better performance
    m.def("forward_wgmma", &flash_look_around_fwd_wgmma_cuda,
          "WGMMA look-around flash attention forward (CUDA, sm_90+)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("proj_weights"), py::arg("causal") = true,
          py::arg("window_left") = -1);

    // SM120 (Blackwell) kernel - uses tcgen05 MMA with TMEM
#if defined(ENABLE_SM120) && ENABLE_SM120
    m.def("forward_sm120", &flash_look_around_fwd_sm120_cuda,
          "SM120 (Blackwell) look-around flash attention forward (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("proj_weights"), py::arg("causal") = true,
          py::arg("window_left") = -1);

    m.def("forward_sm120_pipelined", &flash_look_around_fwd_sm120_pipelined_cuda,
          "SM120 (Blackwell) pipelined look-around flash attention forward (CUDA, 2-stage pipeline)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("proj_weights"), py::arg("causal") = true,
          py::arg("window_left") = -1);

    m.def("forward_sm120_cute", &flash_look_around_fwd_sm120_cute_cuda,
          "SM120 (Blackwell) CuTe hybrid look-around flash attention forward (CuTe TiledMMA + custom softmax/conv)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("proj_weights"), py::arg("causal") = true,
          py::arg("window_left") = -1);
#endif

    // Auto-dispatch forward based on GPU architecture
    m.def("forward_auto", &flash_look_around_fwd_auto_cuda,
          "Auto-dispatch look-around flash attention forward (selects best kernel for GPU)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("proj_weights"), py::arg("causal") = true,
          py::arg("window_left") = -1);

    // Check if SM120 kernel is available
    m.def("is_sm120_available", &is_sm120_kernel_available,
          "Check if SM120 (Blackwell) kernel is available");
}
