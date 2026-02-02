"""Triton kernels for nanochat."""

from nanochat.triton_kernels.look_around_attention import (
    look_around_conv_triton,
    look_around_conv_pytorch,
)
from nanochat.triton_kernels.look_around_v_conv import (
    look_around_v_conv,
    LookAroundVConv,
)

# Try to import kernel strategy functions (only available when Triton is available)
try:
    from nanochat.triton_kernels.look_around_attention import (
        set_triton_kernel_strategy,
        get_triton_kernel_strategy,
        _run_kernel_directly,
    )
except ImportError:
    # Triton not available, provide stubs
    def set_triton_kernel_strategy(use_recompute: bool = True):
        pass

    def get_triton_kernel_strategy() -> str:
        return "pytorch_fallback"

    _run_kernel_directly = None

__all__ = [
    "look_around_conv_triton",
    "look_around_conv_pytorch",
    "set_triton_kernel_strategy",
    "get_triton_kernel_strategy",
    "look_around_v_conv",
    "LookAroundVConv",
]
