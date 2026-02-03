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
from typing import Tuple, Optional

# Try to import the CUDA extension
try:
    import fused_look_around_flash_cuda
    CUDA_KERNEL_AVAILABLE = True
except ImportError:
    CUDA_KERNEL_AVAILABLE = False

if CUDA_KERNEL_AVAILABLE:
    # Register the custom operators with torch.library for torch.compile support
    # This allows the compiler to trace shapes/dtypes via the "Meta" implementation
    # without needing to run the actual CUDA kernel during compilation.

    # 1. Define the operators
    try:
        torch.library.define(
            "nanochat::fused_look_around_flash_fwd",
            "(Tensor q, Tensor k, Tensor v, Tensor proj_weights, bool causal, int window_left) -> (Tensor, Tensor)"
        )
        torch.library.define(
            "nanochat::fused_look_around_flash_bwd",
            "(Tensor grad_out, Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor proj_weights, bool causal, int window_left) -> (Tensor, Tensor, Tensor, Tensor)"
        )
    except Exception:
        # Ignore if already defined (e.g. during reload)
        pass

    # 2. Implement Meta kernels (FakeTensor support for tracing)
    @torch.library.impl_abstract("nanochat::fused_look_around_flash_fwd")
    def _fwd_meta(q, k, v, proj_weights, causal, window_left):
        B, H, T_q, D = q.shape
        # Output has same shape/dtype as q
        out = torch.empty_like(q)
        # LSE (LogSumExp) is (B, H, T_q), typically float32
        lse = torch.empty((B, H, T_q), dtype=torch.float32, device=q.device)
        return out, lse

    @torch.library.impl_abstract("nanochat::fused_look_around_flash_bwd")
    def _bwd_meta(grad_out, q, k, v, out, lse, proj_weights, causal, window_left):
        # Returns gradients for q, k, v, proj_weights
        grad_q = torch.empty_like(q)
        grad_k = torch.empty_like(k)
        grad_v = torch.empty_like(v)
        grad_proj_weights = torch.empty_like(proj_weights)
        return grad_q, grad_k, grad_v, grad_proj_weights

    # 3. Register CUDA implementations
    @torch.library.impl("nanochat::fused_look_around_flash_fwd", "CUDA")
    def _fwd_cuda(q, k, v, proj_weights, causal, window_left):
        return fused_look_around_flash_cuda.forward(q, k, v, proj_weights, causal, window_left)

    @torch.library.impl("nanochat::fused_look_around_flash_bwd", "CUDA")
    def _bwd_cuda(grad_out, q, k, v, out, lse, proj_weights, causal, window_left):
        return fused_look_around_flash_cuda.backward(grad_out, q, k, v, out, lse, proj_weights, causal, window_left)


class FusedLookAroundFlashCUDAFunction(torch.autograd.Function):
    """
    Autograd function for fused look-around flash attention (CUDA implementation).

    This implements the correct backward pass, fixing the bugs in the Triton version:
    1. Proper backward through normalization (D_i = dot(dO, O))
    2. Correct transposed convolution with index shifting
    3. Unified softmax backward after transposed convolution
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,        # (B, H, T_q, D)
        k: torch.Tensor,        # (B, H, T_k, D)
        v: torch.Tensor,        # (B, H, T_k, D)
        proj_logits: torch.Tensor,  # (H, 5) raw logits
        causal: bool = True,
        window_left: int = -1   # -1 for full attention, >= 0 for sliding window
    ) -> torch.Tensor:
        """
        Forward pass of fused look-around flash attention.

        Args:
            q: Query tensor (B, H, T_q, D), will be converted to bfloat16
            k: Key tensor (B, H, T_k, D), will be converted to bfloat16
            v: Value tensor (B, H, T_k, D), will be converted to bfloat16
            proj_logits: Projection logits (H, 5), will be softmaxed internally
            causal: Whether to apply causal masking
            window_left: Sliding window size (-1 for full attention)

        Returns:
            Output tensor (B, H, T_q, D) in bfloat16
        """
        # Ensure inputs are contiguous and in the right dtype
        q = q.contiguous().to(torch.bfloat16)
        k = k.contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.bfloat16)

        # Softmax the projection logits
        proj_weights = F.softmax(proj_logits.float(), dim=-1).contiguous()

        # Call CUDA forward via the registered op
        if CUDA_KERNEL_AVAILABLE:
            out, lse = torch.ops.nanochat.fused_look_around_flash_fwd(
                q, k, v, proj_weights, causal, window_left
            )
        else:
             # Should not happen if check passed before calling apply, but for safety
             raise RuntimeError("CUDA kernel not available")

        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse, proj_weights, proj_logits)
        ctx.causal = causal
        ctx.window_left = window_left

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass with corrected gradient computation.
        """
        q, k, v, out, lse, proj_weights, proj_logits = ctx.saved_tensors
        causal = ctx.causal
        window_left = ctx.window_left

        # Ensure grad_output is contiguous and bfloat16
        grad_output = grad_output.contiguous().to(torch.bfloat16)

        # Call CUDA backward via the registered op
        grad_q, grad_k, grad_v, grad_proj_weights = torch.ops.nanochat.fused_look_around_flash_bwd(
            grad_output, q, k, v, out, lse, proj_weights, causal, window_left
        )

        # Backward through softmax for proj_logits
        # grad_proj_logits = proj_weights * (grad_proj_weights - sum(grad_proj_weights * proj_weights))
        dot_product = (grad_proj_weights * proj_weights).sum(dim=-1, keepdim=True)
        grad_proj_logits = proj_weights * (grad_proj_weights - dot_product)
        grad_proj_logits = grad_proj_logits.to(proj_logits.dtype)

        return grad_q, grad_k, grad_v, grad_proj_logits, None, None


def fused_look_around_flash_attention_cuda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_logits: torch.Tensor,
    causal: bool = True,
    window_left: int = -1,
) -> torch.Tensor:
    """
    Fused look-around flash attention using native CUDA kernel.

    This function applies a 5-tap convolution to attention scores before
    computing the weighted sum of values. It uses a memory-efficient
    flash attention style algorithm that avoids O(N^2) memory usage.

    The convolution formula applied to each row of attention scores:
        P_conv[j] = w0*P[j+2] + w1*P[j+1] + w2*P[j] + w3*P[j-1] + w4*P[j-2]
    where P = softmax(QK^T / sqrt(d_k)) and w = softmax(proj_logits)

    Args:
        q: Query tensor of shape (B, H, T_q, D)
        k: Key tensor of shape (B, H, T_k, D)
        v: Value tensor of shape (B, H, T_k, D)
        proj_logits: Projection logits of shape (H, 5) - will be softmaxed
        causal: Whether to apply causal masking (default True)
        window_left: Sliding window size, -1 for full attention (default -1)

    Returns:
        Output tensor of shape (B, H, T_q, D)

    Raises:
        RuntimeError: If the CUDA kernel is not available (not built)
    """
    if not CUDA_KERNEL_AVAILABLE:
        raise RuntimeError(
            "CUDA kernel not available. Please build with:\n"
            "  cd nanochat/cuda_kernels && pip install -e .\n"
            "Or use the Triton version: "
            "from nanochat.triton_kernels.fused_look_around_flash import fused_look_around_flash_attention"
        )

    return FusedLookAroundFlashCUDAFunction.apply(q, k, v, proj_logits, causal, window_left)


# Convenience alias
fused_look_around_flash = fused_look_around_flash_attention_cuda


def is_cuda_kernel_available() -> bool:
    """Check if the CUDA kernel has been built and is available."""
    return CUDA_KERNEL_AVAILABLE
