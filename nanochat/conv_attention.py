"""
Conv Attention: 1D convolution on attention probabilities after softmax.

  p = softmax(Q @ K^T / sqrt(d))           # one standard softmax (all heads)
  w = softmax(conv_logits)                  # conv kernel, sums to 1
  p_conv[q, k] = sum_j w[j] * p[q, k-j]   # 1D conv along key dim (first n_conv heads)
  p_conv = re_mask(p_conv)                  # zero out causal/window violations
  out = [p_conv; p_std] @ V                 # conv heads ++ standard heads

conv_logits shape (n_conv, 5) determines how many heads get the conv.
Remaining heads are plain attention. If conv_logits is None, all heads standard.
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


def conv_attention_ref(Q, K, V, conv_logits=None, causal=True, window_size=(-1, -1)):
    """
    Reference conv attention: 1D convolution on attention probabilities.

    First n_conv heads get the conv (determined by conv_logits.shape[0]),
    remaining heads get standard attention. If conv_logits is None, all
    heads are standard.

    Args:
        Q, K, V: (B, T, H, D) — FA3 native layout
        conv_logits: (n_conv, 5) or None — raw learnable conv kernel per conv head
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

    # Standard attention logits + mask + ONE softmax (all heads)
    logits = q @ k.transpose(-2, -1) / (D ** 0.5)  # (B, H, T, T)
    mask = _build_mask(T, T, causal, window_size, shift=0, device=Q.device)
    logits = logits.masked_fill(~mask, float('-inf'))
    p = F.softmax(logits, dim=-1)  # (B, H, T, T)
    p = torch.nan_to_num(p, nan=0.0)

    # Apply conv to the first n_conv heads (if any)
    if conv_logits is not None:
        n_conv = conv_logits.shape[0]
        shifts = [-2, -1, 0, 1, 2]
        w = F.softmax(conv_logits, dim=-1)  # (n_conv, 5)

        p_head = p[:, :n_conv]  # (B, n_conv, T, T)
        p_conv = torch.zeros_like(p_head)

        for idx, j in enumerate(shifts):
            shifted_p = torch.zeros_like(p_head)
            if j >= 0:
                if j < T:
                    shifted_p[:, :, :, j:] = p_head[:, :, :, :T - j]
            else:
                if -j < T:
                    shifted_p[:, :, :, :T + j] = p_head[:, :, :, -j:]

            p_conv = p_conv + w[:, idx].view(1, n_conv, 1, 1) * shifted_p

        # Re-mask: conv can push probability to invalid destinations
        p_conv = p_conv.masked_fill(~mask, 0.0)
        p = torch.cat([p_conv, p[:, n_conv:]], dim=1)

    # Attend to original V
    out = p @ v  # (B, H, T, D)
    return out.transpose(1, 2)  # back to (B, T, H, D)
