"""
Fused Look-Around Flash Attention CUDA Kernel

This module provides a native CUDA implementation of fused look-around flash attention
with 5-tap convolution on attention scores. It fixes the backward pass bugs present
in the Triton implementation.

Usage:
    from nanochat.cuda_kernels import fused_look_around_flash_attention_cuda

    # Forward pass
    out = fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal=True)

    # Backward is automatic via autograd
    out.sum().backward()
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional, List

# Try to import the CUDA extension
try:
    import fused_look_around_flash_cuda
    CUDA_KERNEL_AVAILABLE = True
except ImportError:
    CUDA_KERNEL_AVAILABLE = False


# Register custom op for torch.compile compatibility
if CUDA_KERNEL_AVAILABLE:
    # Define the custom op using torch.library
    @torch.library.custom_op("fused_look_around::attention", mutates_args=())
    def _fused_look_around_attention_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        proj_weights: torch.Tensor,
        causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward op - returns (out, lse)"""
        out, lse = fused_look_around_flash_cuda.forward(q, k, v, proj_weights, causal)
        return out, lse

    @_fused_look_around_attention_op.register_fake
    def _fused_look_around_attention_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        proj_weights: torch.Tensor,
        causal: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fake implementation for torch.compile shape inference"""
        B, H, T_q, D = q.shape
        out = q.new_empty((B, H, T_q, D))
        lse = q.new_empty((B, H, T_q), dtype=torch.float32)
        return out, lse

    def _fused_look_around_attention_backward(
        ctx,
        grad_out: torch.Tensor,
        grad_lse: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """Backward implementation for the custom op"""
        q, k, v, out, lse, proj_weights = ctx.saved_tensors
        causal = ctx.causal

        grad_out = grad_out.contiguous().to(torch.bfloat16)

        grad_q, grad_k, grad_v, grad_proj_weights = fused_look_around_flash_cuda.backward(
            grad_out, q, k, v, out, lse, proj_weights, causal
        )

        return grad_q, grad_k, grad_v, grad_proj_weights, None

    def _fused_look_around_attention_setup_context(
        ctx,
        inputs: Tuple,
        output: Tuple[torch.Tensor, torch.Tensor],
    ):
        """Save tensors for backward"""
        q, k, v, proj_weights, causal = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse, proj_weights)
        ctx.causal = causal

    # Register autograd for the custom op
    _fused_look_around_attention_op.register_autograd(
        _fused_look_around_attention_backward,
        setup_context=_fused_look_around_attention_setup_context,
    )


class FusedLookAroundFlashCUDAFunction(torch.autograd.Function):
    """
    Autograd function for fused look-around flash attention (CUDA implementation).

    This is the fallback for when torch.compile is not used.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        proj_logits: torch.Tensor,
        causal: bool = True
    ) -> torch.Tensor:
        q = q.contiguous().to(torch.bfloat16)
        k = k.contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.bfloat16)
        proj_weights = F.softmax(proj_logits.float(), dim=-1).contiguous()

        out, lse = fused_look_around_flash_cuda.forward(q, k, v, proj_weights, causal)

        ctx.save_for_backward(q, k, v, out, lse, proj_weights, proj_logits)
        ctx.causal = causal

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, out, lse, proj_weights, proj_logits = ctx.saved_tensors
        causal = ctx.causal

        grad_output = grad_output.contiguous().to(torch.bfloat16)

        grad_q, grad_k, grad_v, grad_proj_weights = fused_look_around_flash_cuda.backward(
            grad_output, q, k, v, out, lse, proj_weights, causal
        )

        # Backward through softmax for proj_logits
        dot_product = (grad_proj_weights * proj_weights).sum(dim=-1, keepdim=True)
        grad_proj_logits = proj_weights * (grad_proj_weights - dot_product)
        grad_proj_logits = grad_proj_logits.to(proj_logits.dtype)

        return grad_q, grad_k, grad_v, grad_proj_logits, None


def fused_look_around_flash_attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_logits: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Fused look-around flash attention using native CUDA kernel.

    Compatible with torch.compile - uses registered custom op when compiled.

    Args:
        q: Query tensor of shape (B, H, T_q, D)
        k: Key tensor of shape (B, H, T_k, D)
        v: Value tensor of shape (B, H, T_k, D)
        proj_logits: Projection logits of shape (H, 5) - will be softmaxed
        causal: Whether to apply causal masking (default True)

    Returns:
        Output tensor of shape (B, H, T_q, D)
    """
    if not CUDA_KERNEL_AVAILABLE:
        raise RuntimeError(
            "CUDA kernel not available. Please build with:\n"
            "  cd nanochat/cuda_kernels && pip install -e .\n"
            "Or use the Triton version: "
            "from nanochat.triton_kernels.fused_look_around_flash import fused_look_around_flash_attention"
        )

    # Prepare inputs
    q = q.contiguous().to(torch.bfloat16)
    k = k.contiguous().to(torch.bfloat16)
    v = v.contiguous().to(torch.bfloat16)
    proj_weights = F.softmax(proj_logits.float(), dim=-1).contiguous()

    # Use the registered custom op (works with torch.compile)
    out, lse = torch.ops.fused_look_around.attention(q, k, v, proj_weights, causal)

    return out


# Keep the old function as fallback for non-compiled usage
def fused_look_around_flash_attention_cuda_legacy(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_logits: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Legacy version using autograd.Function directly."""
    if not CUDA_KERNEL_AVAILABLE:
        raise RuntimeError("CUDA kernel not available.")
    return FusedLookAroundFlashCUDAFunction.apply(q, k, v, proj_logits, causal)


# Convenience alias
fused_look_around_flash = fused_look_around_flash_attention_cuda


def is_cuda_kernel_available() -> bool:
    """Check if the CUDA kernel has been built and is available."""
    return CUDA_KERNEL_AVAILABLE
