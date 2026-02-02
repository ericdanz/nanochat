// bindings.cpp - PyTorch bindings for fused look-around flash attention

#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
std::vector<torch::Tensor> fused_look_around_flash_fwd_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor proj_weights,
    bool causal
);

std::vector<torch::Tensor> fused_look_around_flash_bwd_cuda(
    torch::Tensor grad_out,
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor lse,
    torch::Tensor proj_weights,
    bool causal
);

// Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_look_around_flash_fwd_cuda,
          "Fused look-around flash attention forward (CUDA)",
          py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("proj_weights"), py::arg("causal") = true);

    m.def("backward", &fused_look_around_flash_bwd_cuda,
          "Fused look-around flash attention backward (CUDA)",
          py::arg("grad_out"), py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("out"), py::arg("lse"), py::arg("proj_weights"),
          py::arg("causal") = true);
}
