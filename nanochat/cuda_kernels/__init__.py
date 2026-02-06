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

# Flag to track if torch.library registration succeeded with correct signature
_TORCH_OPS_AVAILABLE = False

if CUDA_KERNEL_AVAILABLE:
    # Register the custom operators with torch.library for torch.compile support
    # This allows the compiler to trace shapes/dtypes via the "Meta" implementation
    # without needing to run the actual CUDA kernel during compilation.

    # Use versioned operator names to avoid conflicts with stale registrations
    _OP_VERSION = "v2"
    _FWD_OP_NAME = f"nanochat::fused_look_around_flash_fwd_{_OP_VERSION}"
    _BWD_OP_NAME = f"nanochat::fused_look_around_flash_bwd_{_OP_VERSION}"

    # 1. Define the operators
    try:
        torch.library.define(
            _FWD_OP_NAME,
            "(Tensor q, Tensor k, Tensor v, Tensor proj_weights, bool causal, int window_left) -> (Tensor, Tensor)"
        )
        torch.library.define(
            _BWD_OP_NAME,
            "(Tensor grad_out, Tensor q, Tensor k, Tensor v, Tensor out, Tensor lse, Tensor proj_weights, bool causal, int window_left) -> (Tensor, Tensor, Tensor, Tensor)"
        )
        _TORCH_OPS_AVAILABLE = True
    except Exception:
        # Ignore if already defined (e.g. during reload)
        _TORCH_OPS_AVAILABLE = True  # Already registered, assume correct

    if _TORCH_OPS_AVAILABLE:
        # 2. Implement Meta kernels (FakeTensor support for tracing)
        # Supports GQA where Q has more heads than K/V
        try:
            @torch.library.impl_abstract(_FWD_OP_NAME)
            def _fwd_meta(q, k, v, proj_weights, causal, window_left):
                B, n_q_heads, T_q, D = q.shape
                # Output has same shape/dtype as q
                out = torch.empty_like(q)
                # LSE (LogSumExp) is (B, n_q_heads, T_q), typically float32
                lse = torch.empty((B, n_q_heads, T_q), dtype=torch.float32, device=q.device)
                return out, lse

            @torch.library.impl_abstract(_BWD_OP_NAME)
            def _bwd_meta(grad_out, q, k, v, out, lse, proj_weights, causal, window_left):
                # Returns gradients for q, k, v, proj_weights
                # Note: grad_k/grad_v have shape matching k/v (n_kv_heads), not q (n_q_heads)
                grad_q = torch.empty_like(q)
                grad_k = torch.empty_like(k)
                grad_v = torch.empty_like(v)
                grad_proj_weights = torch.empty_like(proj_weights)
                return grad_q, grad_k, grad_v, grad_proj_weights

            # 3. Register CUDA implementations
            @torch.library.impl(_FWD_OP_NAME, "CUDA")
            def _fwd_cuda(q, k, v, proj_weights, causal, window_left):
                return fused_look_around_flash_cuda.forward(q, k, v, proj_weights, causal, window_left)

            @torch.library.impl(_BWD_OP_NAME, "CUDA")
            def _bwd_cuda(grad_out, q, k, v, out, lse, proj_weights, causal, window_left):
                return fused_look_around_flash_cuda.backward(grad_out, q, k, v, out, lse, proj_weights, causal, window_left)
        except Exception:
            _TORCH_OPS_AVAILABLE = False


class FusedLookAroundFlashCUDAFunction(torch.autograd.Function):
    """
    Autograd function for fused look-around flash attention (CUDA implementation).

    This implements the correct backward pass, fixing the bugs in the Triton version:
    1. Proper backward through normalization (D_i = dot(dO, O))
    2. Correct transposed convolution with index shifting
    3. Unified softmax backward after transposed convolution

    Supports GQA (Grouped Query Attention) where n_q_heads > n_kv_heads.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,        # (B, n_q_heads, T_q, D)
        k: torch.Tensor,        # (B, n_kv_heads, T_k, D)
        v: torch.Tensor,        # (B, n_kv_heads, T_k, D)
        proj_logits: torch.Tensor,  # (n_kv_heads, 5) raw logits
        causal: bool = True,
        window_left: int = -1   # -1 for full attention, >= 0 for sliding window
    ) -> torch.Tensor:
        """
        Forward pass of fused look-around flash attention.

        Supports GQA where Q has more heads than K/V. Each group of Q heads
        shares the same K/V head (n_q_heads must be divisible by n_kv_heads).

        Args:
            q: Query tensor (B, n_q_heads, T_q, D), will be converted to bfloat16
            k: Key tensor (B, n_kv_heads, T_k, D), will be converted to bfloat16
            v: Value tensor (B, n_kv_heads, T_k, D), will be converted to bfloat16
            proj_logits: Projection logits (n_kv_heads, 5), will be softmaxed internally
            causal: Whether to apply causal masking
            window_left: Sliding window size (-1 for full attention)

        Returns:
            Output tensor (B, n_q_heads, T_q, D) in bfloat16
        """
        # GQA validation
        n_q_heads = q.shape[1]
        n_kv_heads = k.shape[1]
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads}) for GQA"
            )

        # Ensure inputs are contiguous and in the right dtype
        q = q.contiguous().to(torch.bfloat16)
        k = k.contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.bfloat16)

        # Softmax the projection logits
        proj_weights = F.softmax(proj_logits.float(), dim=-1).contiguous()

        # Call CUDA forward - use torch.ops if available, else direct call
        if _TORCH_OPS_AVAILABLE:
            fwd_op = getattr(torch.ops.nanochat, f"fused_look_around_flash_fwd_{_OP_VERSION}")
            out, lse = fwd_op(q, k, v, proj_weights, causal, window_left)
        elif CUDA_KERNEL_AVAILABLE:
            out, lse = fused_look_around_flash_cuda.forward(q, k, v, proj_weights, causal, window_left)
        else:
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

        # Call CUDA backward - use torch.ops if available, else direct call
        if _TORCH_OPS_AVAILABLE:
            bwd_op = getattr(torch.ops.nanochat, f"fused_look_around_flash_bwd_{_OP_VERSION}")
            grad_q, grad_k, grad_v, grad_proj_weights = bwd_op(
                grad_output, q, k, v, out, lse, proj_weights, causal, window_left
            )
        else:
            grad_q, grad_k, grad_v, grad_proj_weights = fused_look_around_flash_cuda.backward(
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

    Supports GQA (Grouped Query Attention) where Q has more heads than K/V.
    Each group of n_q_heads/n_kv_heads Q heads shares the same K/V head.

    The convolution formula applied to each row of attention scores:
        P_conv[j] = w0*P[j+2] + w1*P[j+1] + w2*P[j] + w3*P[j-1] + w4*P[j-2]
    where P = softmax(QK^T / sqrt(d_k)) and w = softmax(proj_logits)

    Args:
        q: Query tensor of shape (B, n_q_heads, T_q, D)
        k: Key tensor of shape (B, n_kv_heads, T_k, D)
        v: Value tensor of shape (B, n_kv_heads, T_k, D)
        proj_logits: Projection logits of shape (n_kv_heads, 5) - will be softmaxed
        causal: Whether to apply causal masking (default True)
        window_left: Sliding window size, -1 for full attention (default -1)

    Returns:
        Output tensor of shape (B, n_q_heads, T_q, D)

    Raises:
        RuntimeError: If the CUDA kernel is not available (not built)
        ValueError: If n_q_heads is not divisible by n_kv_heads
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


# ============================================================
# FA3-STYLE KERNEL (sm_90+)
# Provides FlashAttention-3 style implementation for Hopper/Blackwell
# ============================================================

# Check if FA3 kernel is available
FA3_KERNEL_AVAILABLE = False
if CUDA_KERNEL_AVAILABLE:
    try:
        FA3_KERNEL_AVAILABLE = fused_look_around_flash_cuda.is_fa3_available()
    except AttributeError:
        FA3_KERNEL_AVAILABLE = False


class FlashLookAroundSM90Function(torch.autograd.Function):
    """
    Autograd function for FA3-style look-around flash attention (sm_90+).

    This uses the new FlashAttention-3 architecture with:
    - Better pipelining and memory access patterns
    - Optimized for Hopper/Blackwell tensor cores
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        proj_logits: torch.Tensor,
        causal: bool = True,
        window_left: int = -1
    ) -> torch.Tensor:
        # GQA validation
        n_q_heads = q.shape[1]
        n_kv_heads = k.shape[1]
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads}) for GQA"
            )

        # Ensure inputs are contiguous and in the right dtype
        q = q.contiguous().to(torch.bfloat16)
        k = k.contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.bfloat16)

        # Softmax the projection logits
        proj_weights = F.softmax(proj_logits.float(), dim=-1).contiguous()

        # Call FA3-style CUDA forward
        out, lse = fused_look_around_flash_cuda.forward_sm90(
            q, k, v, proj_weights, causal, window_left
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse, proj_weights, proj_logits)
        ctx.causal = causal
        ctx.window_left = window_left

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, out, lse, proj_weights, proj_logits = ctx.saved_tensors
        causal = ctx.causal
        window_left = ctx.window_left

        grad_output = grad_output.contiguous().to(torch.bfloat16)

        # Call FA3-style CUDA backward
        grad_q, grad_k, grad_v, grad_proj_weights = fused_look_around_flash_cuda.backward_sm90(
            grad_output, q, k, v, out, lse, proj_weights, causal, window_left
        )

        # Backward through softmax for proj_logits
        dot_product = (grad_proj_weights * proj_weights).sum(dim=-1, keepdim=True)
        grad_proj_logits = proj_weights * (grad_proj_weights - dot_product)
        grad_proj_logits = grad_proj_logits.to(proj_logits.dtype)

        return grad_q, grad_k, grad_v, grad_proj_logits, None, None


def flash_look_around_attention_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_logits: torch.Tensor,
    causal: bool = True,
    window_left: int = -1,
) -> torch.Tensor:
    """
    FA3-style look-around flash attention for sm_90+ GPUs (Hopper/Blackwell).

    This uses the FlashAttention-3 architecture patterns for better performance
    on modern NVIDIA GPUs with WGMMA support.

    Args:
        q: Query tensor of shape (B, n_q_heads, T_q, D)
        k: Key tensor of shape (B, n_kv_heads, T_k, D)
        v: Value tensor of shape (B, n_kv_heads, T_k, D)
        proj_logits: Projection logits of shape (n_kv_heads, 5) - will be softmaxed
        causal: Whether to apply causal masking (default True)
        window_left: Sliding window size, -1 for full attention (default -1)

    Returns:
        Output tensor of shape (B, n_q_heads, T_q, D)

    Raises:
        RuntimeError: If the FA3 kernel is not available (requires sm_90+)
    """
    if not FA3_KERNEL_AVAILABLE:
        raise RuntimeError(
            "FA3-style kernel not available. Requires sm_90+ GPU (Hopper/Blackwell).\n"
            "Use fused_look_around_flash_attention_cuda() for older GPUs."
        )

    return FlashLookAroundSM90Function.apply(q, k, v, proj_logits, causal, window_left)


def fused_look_around_flash_attention_auto(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_logits: torch.Tensor,
    causal: bool = True,
    window_left: int = -1,
) -> torch.Tensor:
    """
    Automatically selects the best kernel based on GPU architecture.

    On sm_90+ (Hopper/Blackwell): Uses FA3-style kernel
    On older GPUs: Uses WMMA-based kernel

    Args:
        q: Query tensor of shape (B, n_q_heads, T_q, D)
        k: Key tensor of shape (B, n_kv_heads, T_k, D)
        v: Value tensor of shape (B, n_kv_heads, T_k, D)
        proj_logits: Projection logits of shape (n_kv_heads, 5) - will be softmaxed
        causal: Whether to apply causal masking (default True)
        window_left: Sliding window size, -1 for full attention (default -1)

    Returns:
        Output tensor of shape (B, n_q_heads, T_q, D)
    """
    if FA3_KERNEL_AVAILABLE:
        return flash_look_around_attention_sm90(q, k, v, proj_logits, causal, window_left)
    else:
        return fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal, window_left)


def is_fa3_kernel_available() -> bool:
    """Check if the FA3-style kernel is available (requires sm_90+ GPU)."""
    return FA3_KERNEL_AVAILABLE


# ============================================================
# SM120 (BLACKWELL) KERNEL
# Uses tcgen05 MMA instructions with TMEM accumulator
# ============================================================

# Check if SM120 kernel is available
SM120_KERNEL_AVAILABLE = False
if CUDA_KERNEL_AVAILABLE:
    try:
        SM120_KERNEL_AVAILABLE = fused_look_around_flash_cuda.is_sm120_available()
    except AttributeError:
        SM120_KERNEL_AVAILABLE = False


class FlashLookAroundSM120Function(torch.autograd.Function):
    """
    Autograd function for SM120 (Blackwell) look-around flash attention.

    This uses tcgen05 MMA instructions with TMEM accumulator for
    optimal performance on Blackwell GPUs.
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        proj_logits: torch.Tensor,
        causal: bool = True,
        window_left: int = -1
    ) -> torch.Tensor:
        # GQA validation
        n_q_heads = q.shape[1]
        n_kv_heads = k.shape[1]
        if n_q_heads % n_kv_heads != 0:
            raise ValueError(
                f"n_q_heads ({n_q_heads}) must be divisible by n_kv_heads ({n_kv_heads}) for GQA"
            )

        # Ensure inputs are contiguous and in the right dtype
        q = q.contiguous().to(torch.bfloat16)
        k = k.contiguous().to(torch.bfloat16)
        v = v.contiguous().to(torch.bfloat16)

        # Softmax the projection logits
        proj_weights = F.softmax(proj_logits.float(), dim=-1).contiguous()

        # Call SM120 CUDA forward
        out, lse = fused_look_around_flash_cuda.forward_sm120(
            q, k, v, proj_weights, causal, window_left
        )

        # Save for backward
        ctx.save_for_backward(q, k, v, out, lse, proj_weights, proj_logits)
        ctx.causal = causal
        ctx.window_left = window_left

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, out, lse, proj_weights, proj_logits = ctx.saved_tensors
        causal = ctx.causal
        window_left = ctx.window_left

        grad_output = grad_output.contiguous().to(torch.bfloat16)

        # Use SM90 backward for now (SM120 backward not yet implemented)
        grad_q, grad_k, grad_v, grad_proj_weights = fused_look_around_flash_cuda.backward_sm90(
            grad_output, q, k, v, out, lse, proj_weights, causal, window_left
        )

        # Backward through softmax for proj_logits
        dot_product = (grad_proj_weights * proj_weights).sum(dim=-1, keepdim=True)
        grad_proj_logits = proj_weights * (grad_proj_weights - dot_product)
        grad_proj_logits = grad_proj_logits.to(proj_logits.dtype)

        return grad_q, grad_k, grad_v, grad_proj_logits, None, None


def flash_look_around_attention_sm120(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_logits: torch.Tensor,
    causal: bool = True,
    window_left: int = -1,
) -> torch.Tensor:
    """
    SM120 (Blackwell) look-around flash attention using tcgen05/TMEM.

    This uses Blackwell's new tensor core instructions for optimal performance:
    - tcgen05.mma instructions with TMEM accumulator
    - 256KB TMEM per SM for accumulator storage
    - 3-stage pipelining for better memory hiding

    Args:
        q: Query tensor of shape (B, n_q_heads, T_q, D)
        k: Key tensor of shape (B, n_kv_heads, T_k, D)
        v: Value tensor of shape (B, n_kv_heads, T_k, D)
        proj_logits: Projection logits of shape (n_kv_heads, 5) - will be softmaxed
        causal: Whether to apply causal masking (default True)
        window_left: Sliding window size, -1 for full attention (default -1)

    Returns:
        Output tensor of shape (B, n_q_heads, T_q, D)

    Raises:
        RuntimeError: If the SM120 kernel is not available (requires Blackwell GPU)
    """
    if not SM120_KERNEL_AVAILABLE:
        raise RuntimeError(
            "SM120 kernel not available. Requires Blackwell GPU (sm_120) with CUDA 13.0+.\n"
            "Use fused_look_around_flash_attention_cuda() or flash_look_around_attention_sm90() instead."
        )

    return FlashLookAroundSM120Function.apply(q, k, v, proj_logits, causal, window_left)


def is_sm120_kernel_available() -> bool:
    """Check if the SM120 (Blackwell) kernel is available."""
    return SM120_KERNEL_AVAILABLE


# Update auto-dispatch to include SM120
def fused_look_around_flash_attention_best(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_logits: torch.Tensor,
    causal: bool = True,
    window_left: int = -1,
) -> torch.Tensor:
    """
    Automatically selects the best kernel based on GPU architecture.

    Priority order:
    1. SM120 (Blackwell) - if available, uses tcgen05/TMEM
    2. FA3/WGMMA (Hopper) - if sm_90+ detected
    3. WMMA (fallback) - for older GPUs

    Args:
        q: Query tensor of shape (B, n_q_heads, T_q, D)
        k: Key tensor of shape (B, n_kv_heads, T_k, D)
        v: Value tensor of shape (B, n_kv_heads, T_k, D)
        proj_logits: Projection logits of shape (n_kv_heads, 5) - will be softmaxed
        causal: Whether to apply causal masking (default True)
        window_left: Sliding window size, -1 for full attention (default -1)

    Returns:
        Output tensor of shape (B, n_q_heads, T_q, D)
    """
    if SM120_KERNEL_AVAILABLE:
        return flash_look_around_attention_sm120(q, k, v, proj_logits, causal, window_left)
    elif FA3_KERNEL_AVAILABLE:
        return flash_look_around_attention_sm90(q, k, v, proj_logits, causal, window_left)
    else:
        return fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal, window_left)
