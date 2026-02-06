"""
Conv Attention: 1D convolution on attention patterns via mixture of shifted softmaxes.

For conv kernel w = softmax(conv_logits) with shifts j in {-2, -1, 0, 1, 2}:
  For each shift j:
    shifted_logits[q, k] = logits[q, k+j]  (-inf where k+j is invalid)
    p_j = softmax(shifted_logits)
    shifted_V[k] = V[k+j]                  (0 where k+j is invalid)
    out_j = p_j @ shifted_V
  Final: out = sum_j w[j] * out_j

Each p_j is a proper probability distribution. w sums to 1. Result is a convex
combination of valid attended values with no ad-hoc renormalization.
"""
import torch
import torch.nn.functional as F


def standard_attention_ref(Q, K, V, causal=True, window_size=(-1, -1)):
    """
    Standard scaled dot-product attention (reference).

    Args:
        Q, K, V: (B, T, H, D) — FA3 native layout
        causal: bool
        window_size: (left, right) tuple, -1 = unlimited
    Returns:
        (B, T, H, D)
    """
    B, T, H, D = Q.shape
    # Transpose to (B, H, T, D) for matmul
    q = Q.transpose(1, 2)
    k = K.transpose(1, 2)
    v = V.transpose(1, 2)

    logits = q @ k.transpose(-2, -1) / (D ** 0.5)  # (B, H, T, T)

    mask = _build_mask(T, T, causal, window_size, shift=0, device=Q.device)
    logits = logits.masked_fill(~mask, float('-inf'))

    p = F.softmax(logits, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)

    out = p @ v  # (B, H, T, D)
    return out.transpose(1, 2)


def _build_mask(Tq, Tk, causal, window_size, shift, device):
    """
    Build a (Tq, Tk) boolean mask for shifted attention.

    For shift j, the effective key position is k+j. We need:
      - 0 <= k+j < Tk  (in-bounds)
      - If causal: k+j <= q
      - If windowed (left): k+j >= q - window_left
      - If windowed (right): k+j <= q + window_right (only if not causal)

    Since logits[q, k] corresponds to key position k+j after shifting:
      - k ranges over [0, Tk), but k+j must be in [0, Tk)
      - So k must be in [max(0, -j), min(Tk, Tk-j))
      - Causal: k + j <= q  =>  k <= q - j
      - Window left: k + j >= q - W_left  =>  k >= q - W_left - j

    Args:
        Tq, Tk: query/key sequence lengths
        causal: bool
        window_size: (left, right), -1 = unlimited
        shift: integer shift j
        device: torch device
    Returns:
        (Tq, Tk) bool tensor
    """
    q_idx = torch.arange(Tq, device=device).unsqueeze(1)  # (Tq, 1)
    k_idx = torch.arange(Tk, device=device).unsqueeze(0)  # (1, Tk)

    effective_k = k_idx + shift  # effective key position

    # In-bounds check
    mask = (effective_k >= 0) & (effective_k < Tk)

    if causal:
        mask = mask & (effective_k <= q_idx)

    w_left, w_right = window_size
    if w_left >= 0:
        mask = mask & (effective_k >= q_idx - w_left)
    if w_right >= 0 and not causal:
        mask = mask & (effective_k <= q_idx + w_right)

    return mask


def conv_attention_ref(Q, K, V, conv_logits, causal=True, window_size=(-1, -1)):
    """
    Reference conv attention (materializes full attention matrix).

    Args:
        Q, K, V: (B, T, H, D) — FA3 native layout
        conv_logits: (H, 5) — raw learnable conv kernel per head
        causal: bool
        window_size: (left, right) tuple, -1 = unlimited
    Returns:
        (B, T, H, D)
    """
    B, T, H, D = Q.shape
    shifts = [-2, -1, 0, 1, 2]

    # Transpose to (B, H, T, D) for matmul
    q = Q.transpose(1, 2)
    k = K.transpose(1, 2)
    v = V.transpose(1, 2)

    # Base logits: (B, H, T, T)
    logits = q @ k.transpose(-2, -1) / (D ** 0.5)

    # Conv kernel weights: softmax over the 5 shifts per head
    w = F.softmax(conv_logits, dim=-1)  # (H, 5)

    out = torch.zeros_like(q)  # (B, H, T, D)

    for idx, j in enumerate(shifts):
        # Build shifted logits by slicing (no wrapping)
        shifted_logits = torch.full_like(logits, float('-inf'))

        # logits[q, k] with effective key k+j means we read from logits[:, :, q, k+j]
        # For shifted_logits[q, k] = logits[q, k+j]:
        #   source column: k+j, so source range [max(0,j), T+min(0,j)) in source
        #   dest column:   k,   so dest range   [max(0,-j), T+min(0,-j)) in dest
        if j >= 0:
            src_start, src_end = j, T
            dst_start, dst_end = 0, T - j
        else:
            src_start, src_end = 0, T + j
            dst_start, dst_end = -j, T

        shifted_logits[:, :, :, dst_start:dst_end] = logits[:, :, :, src_start:src_end]

        # Apply causal + window mask
        mask = _build_mask(T, T, causal, window_size, shift=j, device=Q.device)
        shifted_logits = shifted_logits.masked_fill(~mask, float('-inf'))

        # Softmax over keys, nan_to_num for all-masked rows
        p_j = F.softmax(shifted_logits, dim=-1)
        p_j = torch.nan_to_num(p_j, nan=0.0)

        # Build shifted V: shifted_V[k] = V[k+j], 0 where out of bounds
        shifted_v = torch.zeros_like(v)
        if j >= 0:
            shifted_v[:, :, :T - j, :] = v[:, :, j:, :]
        else:
            shifted_v[:, :, -j:, :] = v[:, :, :T + j, :]

        out_j = p_j @ shifted_v  # (B, H, T, D)

        # Weight by conv kernel: w[h, idx] broadcast over (B, H, T, D)
        out = out + w[:, idx].view(1, H, 1, 1) * out_j

    return out.transpose(1, 2)  # back to (B, T, H, D)
