"""
Conv Attention: 1D convolution on attention probabilities after softmax.

  p = softmax(Q @ K^T / sqrt(d))           # one standard softmax (all heads)
  w = softmax(conv_logits)                  # conv kernel, sums to 1
  p_conv[q, k] = sum_j w[j] * p[q, k-j]   # 1D conv along key dim (first n_conv heads)
  p_conv = re_mask(p_conv)                  # zero out causal/window violations
  out = [p_conv; p_std] @ V                 # conv heads ++ standard heads

conv_logits shape (n_conv, 5) determines how many heads get the conv.
Remaining heads are plain attention. If conv_logits is None, all heads standard.

Fast path (conv_attention_fast): pre-convolve V, run unmodified FA3, correct boundaries.
"""
import torch
import torch.nn.functional as F

from nanochat.flash_attention import flash_attn


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


# =============================================================================
# Fast path: pre-convolve V, run FA3, correct boundaries
# =============================================================================

def _convolve_v(V, w):
    """Pre-convolve V along the sequence dimension with conv weights.

    Uses F.conv1d with grouped convolution for efficiency.

    Args:
        V: (B, T, n_conv, D)
        w: (n_conv, 5) softmaxed weights for shifts [-2, -1, 0, 1, 2]
    Returns:
        V_conv: (B, T, n_conv, D) where V_conv[m] = Σ_j w[j] * V[m + shift_j]
    """
    B, T, n_conv, D = V.shape
    # Reshape to (B*D, n_conv, T) for grouped conv1d
    x = V.permute(0, 3, 2, 1).reshape(B * D, n_conv, T)
    # F.conv1d cross-correlation with padding=2:
    #   out[t] = Σ_k weight[k] * input[t + k - 2]
    # which gives V_conv[m] = Σ_j w[j] * V[m + (j-2)] = Σ_j w[j] * V[m + shift_j]
    kernel = w.unsqueeze(1)  # (n_conv, 1, 5)
    out = F.conv1d(x, kernel, padding=2, groups=n_conv)  # (B*D, n_conv, T)
    return out.reshape(B, D, n_conv, T).permute(0, 3, 2, 1)  # (B, T, n_conv, D)


def _apply_causal_correction(out, Q, K, V, w, lse, n_conv):
    """Subtract causal boundary leakage from positive shifts.

    Positive shifts (j=3: shift +1, j=4: shift +2) in V_conv let query q
    "see" V[q+1] and V[q+2] via convolved V at positions m <= q.
    The reference re-masks these out. We subtract them here.

    correction[q] = w[3] * p[q,q] * V[q+1]           # shift +1, m=q
                  + w[4] * p[q,q-1] * V[q+1]          # shift +2, m=q-1
                  + w[4] * p[q,q] * V[q+2]            # shift +2, m=q

    Args:
        out: (B, T, H, D) — FA3 output to correct (in-place on conv heads)
        Q, K, V: (B, T, H, D) — original inputs
        w: (n_conv, 5) — softmaxed conv weights
        lse: (B, H, T) — log-sum-exp from FA3 (detached)
        n_conv: number of conv heads
    Returns:
        corrected out: (B, T, H, D)
    """
    B, T, H, D = Q.shape
    out_dtype = out.dtype
    scale = D ** (-0.5)

    Qc = Q[:, :, :n_conv, :]   # (B, T, n_conv, D)
    Kc = K[:, :, :n_conv, :]
    Vc = V[:, :, :n_conv, :]
    lse_t = lse[:, :n_conv, :].transpose(1, 2)  # (B, T, n_conv)

    w3 = w[:, 3].to(out_dtype).view(1, 1, n_conv, 1)
    w4 = w[:, 4].to(out_dtype).view(1, 1, n_conv, 1)

    # p[q,q] = exp(Q[q]·K[q]*scale - LSE[q])
    p_qq = torch.exp(((Qc * Kc).sum(-1) * scale - lse_t).to(out_dtype)).unsqueeze(-1)

    # V[q+1] (zero-padded at end) and V[q+2]
    V_qp1 = F.pad(Vc[:, 1:], (0, 0, 0, 0, 0, 1))  # (B, T, n_conv, D)
    V_qp2 = F.pad(Vc[:, 2:], (0, 0, 0, 0, 0, 2)) if T > 2 else torch.zeros_like(Vc)

    # correction = w3*p[q,q]*V[q+1] + w4*p[q,q]*V[q+2]
    correction = w3 * p_qq * V_qp1 + w4 * p_qq * V_qp2

    # + w4*p[q,q-1]*V[q+1] (for q >= 1)
    if T > 1:
        p_qm1 = torch.exp(((Qc[:, 1:] * Kc[:, :T-1]).sum(-1) * scale - lse_t[:, 1:]).to(out_dtype))
        p_qm1 = F.pad(p_qm1, (0, 0, 1, 0)).unsqueeze(-1)  # zero at q=0
        correction = correction + w4 * p_qm1 * V_qp1

    out = out.clone()
    out[:, :, :n_conv, :] = out[:, :, :n_conv, :] - correction

    return out


def _apply_window_left_correction(out, Q, K, V, w, lse, n_conv, window_left):
    """Subtract left-window boundary leakage from negative shifts.

    Negative shifts (j=0: shift -2, j=1: shift -1) in V_conv let query q
    access V[q-W-1] and V[q-W-2] via convolved V at positions m within
    the window [q-W, q]. The reference re-masks these out.

    correction[q] = w[1] * p[q, q-W] * V[q-W-1]           # shift -1, m=q-W
                  + w[0] * p[q, q-W+1] * V[q-W-1]          # shift -2, m=q-W+1
                  + w[0] * p[q, q-W] * V[q-W-2]            # shift -2, m=q-W

    Only applies when q > W (otherwise the window doesn't clip).

    Args:
        out: (B, T, H, D)
        Q, K, V: (B, T, H, D)
        w: (n_conv, 5) softmaxed conv weights
        lse: (B, H, T) detached log-sum-exp from FA3
        n_conv: number of conv heads
        window_left: int, the left window size W
    Returns:
        corrected out: (B, T, H, D)
    """
    B, T, H, D = Q.shape
    out_dtype = out.dtype
    W = window_left
    scale = D ** (-0.5)

    Qc = Q[:, :, :n_conv, :]
    Kc = K[:, :, :n_conv, :]
    Vc = V[:, :, :n_conv, :]
    lse_c = lse[:, :n_conv, :]  # (B, n_conv, T)

    w0 = w[:, 0].to(out_dtype).view(1, 1, n_conv, 1)  # shift -2
    w1 = w[:, 1].to(out_dtype).view(1, 1, n_conv, 1)  # shift -1

    if T <= W:
        return out  # window never clips

    q_start = W + 1
    if q_start >= T:
        return out

    n_active = T - q_start

    # Q at active positions, K at boundary positions
    Q_active = Qc[:, q_start:]          # (B, n_active, n_conv, D)
    K_boundary = Kc[:, 1:T-W]           # K[q-W] for q in [q_start, T)
    lse_active = lse_c[:, :, q_start:].transpose(1, 2)  # (B, n_active, n_conv)

    # p[q, q-W]
    p_boundary = torch.exp(
        ((Q_active * K_boundary).sum(-1) * scale - lse_active).to(out_dtype)
    ).unsqueeze(-1)

    # V[q-W-1] for q in [q_start, T): V[0:T-W-1]
    V_bm1 = Vc[:, 0:T-W-1]

    # Shift -1: w1 * p[q, q-W] * V[q-W-1]
    corr = w1 * p_boundary * V_bm1

    # Shift -2, term 1: w0 * p[q, q-W] * V[q-W-2]
    V_bm2 = F.pad(Vc[:, 0:T-W-2], (0, 0, 0, 0, 1, 0)) if n_active > 1 else torch.zeros_like(V_bm1)
    corr = corr + w0 * p_boundary * V_bm2

    # Shift -2, term 2: w0 * p[q, q-W+1] * V[q-W-1]
    K_bp1 = Kc[:, 2:T-W+1]  # K[q-W+1]
    p_bp1 = torch.exp(
        ((Q_active * K_bp1).sum(-1) * scale - lse_active).to(out_dtype)
    ).unsqueeze(-1)
    corr = corr + w0 * p_bp1 * V_bm1

    out = out.clone()
    out[:, q_start:, :n_conv, :] = out[:, q_start:, :n_conv, :] - corr

    return out


def conv_attention_fast(Q, K, V, conv_logits=None, causal=True, window_size=(-1, -1)):
    """Fast conv attention: pre-convolve V, run FA3, correct boundaries.

    Mathematically equivalent to conv_attention_ref but runs at FA3 speed
    since the kernel is unmodified. Only the V pre-convolution (O(T·D·5))
    and boundary corrections (O(T·D)) are added.

    Args:
        Q, K, V: (B, T, H, D) — FA3 native layout
        conv_logits: (n_conv, 5) or None — raw learnable conv kernel per conv head
        causal: bool
        window_size: (left, right) tuple, -1 = unlimited
    Returns:
        (B, T, H, D)
    """
    if conv_logits is None:
        return flash_attn.flash_attn_func(Q, K, V, causal=causal, window_size=window_size)

    n_conv = conv_logits.shape[0]
    w = F.softmax(conv_logits.float(), dim=-1)  # (n_conv, 5) always float32

    # Pre-convolve V for conv heads (cast w to V's dtype for FA3 compatibility)
    V_conv = _convolve_v(V[:, :, :n_conv, :], w.to(V.dtype))
    V_combined = torch.cat([V_conv, V[:, :, n_conv:, :]], dim=2)

    # Standard FA3 with convolved V — returns (out, softmax_lse)
    out, lse = flash_attn.flash_attn_func(Q, K, V_combined, causal=causal,
                                           window_size=window_size, return_attn_probs=True)

    # Boundary corrections (subtract leakage from positive/negative shifts)
    # Corrections computed in float32 for precision, cast to output dtype
    if causal:
        out = _apply_causal_correction(out, Q, K, V, w, lse.detach(), n_conv)
    if window_size[0] >= 0:
        out = _apply_window_left_correction(out, Q, K, V, w, lse.detach(), n_conv, window_size[0])

    return out
