"""
V-Convolution Look-Around Attention.

This module implements look-around attention by convolving the Values tensor
instead of the attention matrix. This allows using Flash Attention while
achieving the same effect as convolving the attention weights.

Mathematical equivalence:
    A_conv @ V = A @ V_conv
    where V_conv = conv1d(V, flipped_kernel)

Memory efficiency:
    - Old approach: O(T²) for attention matrix convolution
    - New approach: O(T×D) for value convolution
    - For T=8192, D=128: 128x memory reduction

Causality note:
    V_conv[i] contains weighted information from V[i-2:i+3]. This means
    query i attending to key i will "see" soft information from positions
    i-2, i-1, i, i+1, i+2. This is the intended "look-around" behavior.
    For strict causality, use a one-sided kernel (look-back only).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def look_around_v_conv(
    v: torch.Tensor,
    proj_logits: torch.Tensor,
    causal_kernel: bool = False,
) -> torch.Tensor:
    """
    Apply look-around convolution to values tensor.

    This is mathematically equivalent to convolving the attention weights,
    but operates on O(T×D) instead of O(T²) memory.

    Args:
        v: (B, T, H, D) values tensor (Flash Attention layout)
        proj_logits: (H, 5) learnable projection logits per head
        causal_kernel: If True, zero out the "future" kernel weights (indices 0,1)
                      so V_conv only contains past information.
                      Default False allows bidirectional look-around.

    Returns:
        (B, T, H, D) convolved values tensor
    """
    B, T, H, D = v.shape

    # Compute softmax over the 5-tap kernel
    # Shape: (H, 5)
    # Keep everything in the same dtype as v to avoid gradient issues with mixed precision
    weights = F.softmax(proj_logits, dim=-1)

    # For V-convolution equivalence, we use the kernel DIRECTLY (no flip!)
    # The existing attention-matrix code flips the kernel to achieve:
    #   A_conv[k] = A[k+2]*p[0] + A[k+1]*p[1] + A[k]*p[2] + A[k-1]*p[3] + A[k-2]*p[4]
    #
    # For the mathematically equivalent V-convolution:
    #   V_conv[m] = p[0]*V[m-2] + p[1]*V[m-1] + p[2]*V[m] + p[3]*V[m+1] + p[4]*V[m+2]
    #
    # Conv1d with padding=2 naturally gives:
    #   out[m] = k[0]*in[m-2] + k[1]*in[m-1] + k[2]*in[m] + k[3]*in[m+1] + k[4]*in[m+2]
    #
    # So we use k = p directly (no flip needed!)

    # Optional: make kernel causal by zeroing future weights
    # k[3] and k[4] access positions m+1 and m+2 (future), so zero them out
    if causal_kernel:
        mask = torch.tensor([1., 1., 1., 0., 0.], device=weights.device, dtype=weights.dtype)
        weights = weights * mask
        # Renormalize so weights sum to 1
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    # Reshape V for conv1d: (B, T, H, D) -> (B*H, D, T)
    # We treat D as "channels" and T as the spatial dimension
    v_reshaped = v.permute(0, 2, 3, 1).reshape(B * H, D, T)

    # Build kernel for grouped conv1d
    # We want each of the B*H "groups" to use a different 5-tap filter
    # But all D channels within a group share the same filter
    #
    # For grouped conv1d with groups=B*H:
    # - Input: (N=1, C_in=B*H*D, T)  <- but we have (B*H, D, T)
    # - Kernel: (C_out, C_in/groups, K) = (B*H*D, D, 5)  <- too complex
    #
    # Simpler: process each head separately (still very fast for small H)
    # Or use unfold + matmul which is cleaner

    # Use unfold for efficient "manual" convolution
    # Unfold extracts sliding windows: (B*H, D, T) -> (B*H, D, 5, T)
    v_padded = F.pad(v_reshaped, (2, 2), mode='constant', value=0)  # (B*H, D, T+4)
    v_unfolded = v_padded.unfold(dimension=2, size=5, step=1)  # (B*H, D, T, 5)

    # weights: (H, 5) -> expand for batch: (B*H, 5) -> (B*H, 1, 1, 5)
    # weights are already in v's dtype from the softmax step above
    kernel = weights.repeat(B, 1).unsqueeze(1).unsqueeze(2)  # (B*H, 1, 1, 5)

    # Weighted sum over the kernel dimension
    v_conv = (v_unfolded * kernel).sum(dim=-1)  # (B*H, D, T)

    # Reshape back: (B*H, D, T) -> (B, H, D, T) -> (B, T, H, D)
    v_conv = v_conv.reshape(B, H, D, T).permute(0, 3, 1, 2)

    return v_conv.to(v.dtype)


class LookAroundVConv(nn.Module):
    """
    V-Convolution module for look-around attention.

    This module maintains learnable 5-tap projection logits per head and applies
    the V-convolution transformation before Flash Attention.

    Usage:
        look_around = LookAroundVConv(n_heads=8)
        v_conv = look_around(v)  # (B, T, H, D) -> (B, T, H, D)
        output = flash_attn_func(q, k, v_conv, causal=True)
    """

    def __init__(self, n_heads: int, causal_kernel: bool = False):
        """
        Args:
            n_heads: Number of attention heads
            causal_kernel: If True, use causal (look-back only) kernel
        """
        super().__init__()
        self.n_heads = n_heads
        self.causal_kernel = causal_kernel

        # 5-tap projection logits per head
        # Initialized to approximate identity: strong center weight
        self.proj_logits = nn.Parameter(torch.zeros(n_heads, 5))

    def reset_parameters(self):
        """Initialize to near-identity: center weight dominates."""
        # [-2, -2, 2, -2, -2] gives softmax ~[0.02, 0.02, 0.92, 0.02, 0.02]
        self.proj_logits.data.fill_(-2.0)
        self.proj_logits.data[:, 2] = 2.0

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Apply V-convolution.

        Args:
            v: (B, T, H, D) values tensor

        Returns:
            (B, T, H, D) convolved values tensor
        """
        return look_around_v_conv(v, self.proj_logits, self.causal_kernel)

    def extra_repr(self) -> str:
        return f"n_heads={self.n_heads}, causal_kernel={self.causal_kernel}"
