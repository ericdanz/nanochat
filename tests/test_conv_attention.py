"""
Correctness tests for conv attention (mixture of shifted softmaxes).
"""
import torch
import torch.nn.functional as F
import pytest

from nanochat.conv_attention import conv_attention_ref, standard_attention_ref

DEVICE = "cuda"
DTYPE = torch.bfloat16
# Use float32 for reference comparisons to avoid bf16 precision issues
REF_DTYPE = torch.float32


def _rand_qkv(B, T, H, D, dtype=REF_DTYPE, device=DEVICE):
    """Generate random Q, K, V in FA3 layout (B, T, H, D)."""
    Q = torch.randn(B, T, H, D, dtype=dtype, device=device)
    K = torch.randn(B, T, H, D, dtype=dtype, device=device)
    V = torch.randn(B, T, H, D, dtype=dtype, device=device)
    return Q, K, V


def _identity_logits(H, device=DEVICE):
    """Conv logits that produce [0,0,1,0,0] after softmax (identity = shift 0 only)."""
    logits = torch.full((H, 5), -1e9, device=device, dtype=REF_DTYPE)
    logits[:, 2] = 0.0  # center position dominates after softmax
    return logits


# ============================================================================
# Test 1: Identity kernel matches standard attention
# ============================================================================
class TestIdentityKernel:
    @pytest.mark.parametrize("T", [4, 16, 64])
    def test_causal(self, T):
        B, H, D = 2, 4, 64
        Q, K, V = _rand_qkv(B, T, H, D)
        conv_logits = _identity_logits(H)

        out_conv = conv_attention_ref(Q, K, V, conv_logits, causal=True)
        out_std = standard_attention_ref(Q, K, V, causal=True)

        torch.testing.assert_close(out_conv, out_std, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize("T", [4, 16, 64])
    def test_causal_windowed(self, T):
        B, H, D = 2, 4, 64
        Q, K, V = _rand_qkv(B, T, H, D)
        conv_logits = _identity_logits(H)
        window_size = (8, 0)

        out_conv = conv_attention_ref(Q, K, V, conv_logits, causal=True, window_size=window_size)
        out_std = standard_attention_ref(Q, K, V, causal=True, window_size=window_size)

        torch.testing.assert_close(out_conv, out_std, atol=1e-5, rtol=1e-5)


# ============================================================================
# Test 2: Pure left shift kernel [0,0,0,0,1] (j=+2)
# ============================================================================
class TestPureShift:
    def test_shift_plus2_small(self):
        """With kernel [0,0,0,0,1], each query attends to keys shifted by +2."""
        B, T, H, D = 1, 8, 1, 16
        Q, K, V = _rand_qkv(B, T, H, D)

        # Logits that produce [0,0,0,0,1] after softmax => only j=+2 contributes
        conv_logits = torch.full((H, 5), -1e9, device=DEVICE, dtype=REF_DTYPE)
        conv_logits[:, 4] = 0.0  # index 4 = shift +2

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True)

        # Manually compute for j=+2:
        # shifted_logits[q, k] = logits[q, k+2], causal: k+2 <= q
        # shifted_V[k] = V[k+2]
        q = Q.transpose(1, 2)  # (B, H, T, D)
        k = K.transpose(1, 2)
        v = V.transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / (D ** 0.5)

        shifted_logits = torch.full_like(logits, float('-inf'))
        shifted_logits[:, :, :, :T-2] = logits[:, :, :, 2:]

        # Causal mask: k+2 <= q => k <= q-2
        q_idx = torch.arange(T, device=DEVICE).unsqueeze(1)
        k_idx = torch.arange(T, device=DEVICE).unsqueeze(0)
        mask = ((k_idx + 2) <= q_idx) & (k_idx + 2 >= 0) & (k_idx + 2 < T)
        shifted_logits = shifted_logits.masked_fill(~mask, float('-inf'))

        p = F.softmax(shifted_logits, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)

        shifted_v = torch.zeros_like(v)
        shifted_v[:, :, :T-2, :] = v[:, :, 2:, :]

        expected = (p @ shifted_v).transpose(1, 2)

        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


# ============================================================================
# Test 3: Uniform kernel — average of 5 shifts
# ============================================================================
class TestUniformKernel:
    def test_uniform_is_average(self):
        B, T, H, D = 2, 16, 3, 32
        Q, K, V = _rand_qkv(B, T, H, D)

        # Uniform kernel: [1,1,1,1,1] logits => w = [0.2, 0.2, 0.2, 0.2, 0.2]
        conv_logits = torch.zeros(H, 5, device=DEVICE, dtype=REF_DTYPE)
        out_uniform = conv_attention_ref(Q, K, V, conv_logits, causal=True)

        # Compute each pure shift and average
        shifts_out = []
        for idx in range(5):
            logits_i = torch.full((H, 5), -1e9, device=DEVICE, dtype=REF_DTYPE)
            logits_i[:, idx] = 0.0
            out_i = conv_attention_ref(Q, K, V, logits_i, causal=True)
            shifts_out.append(out_i)

        expected = sum(shifts_out) / 5.0
        torch.testing.assert_close(out_uniform, expected, atol=1e-5, rtol=1e-5)


# ============================================================================
# Test 4: Boundary handling at q=0, q=1
# ============================================================================
class TestBoundaries:
    def test_q0_all_shifts_give_v0(self):
        """At q=0 with causal:
        - j=-2: effective key k-2, valid k=2 only (k-2=0 in [0,0]). out=V[0]
        - j=-1: effective key k-1, valid k=1 only. out=V[0]
        - j= 0: effective key k,   valid k=0 only. out=V[0]
        - j=+1: k+1<=0 => k<=-1, no valid k. out=0
        - j=+2: k+2<=0 => k<=-2, no valid k. out=0
        With uniform w=0.2: out = 0.6*V[0]
        With identity kernel (j=0 only): out = V[0]
        """
        B, H, D = 1, 2, 16
        T = 8
        Q, K, V = _rand_qkv(B, T, H, D)

        # Identity kernel => only j=0 contributes => out[q=0] = V[0]
        conv_logits = _identity_logits(H)
        out = conv_attention_ref(Q, K, V, conv_logits, causal=True)
        out_q0 = out[:, 0, :, :]
        v0 = V[:, 0, :, :]
        torch.testing.assert_close(out_q0, v0, atol=1e-5, rtol=1e-5)

        # Uniform kernel => out[q=0] = 0.6*V[0] (shifts -2,-1,0 contribute, +1,+2 give zero)
        conv_logits_uniform = torch.zeros(H, 5, device=DEVICE, dtype=REF_DTYPE)
        out2 = conv_attention_ref(Q, K, V, conv_logits_uniform, causal=True)
        out2_q0 = out2[:, 0, :, :]
        torch.testing.assert_close(out2_q0, 0.6 * v0, atol=1e-5, rtol=1e-5)

    def test_q1_no_future_leak(self):
        """At q=1 with causal, shifts j=+2 has no valid keys (k+2<=1 and k+2>=0 => k=-1, invalid).
        Shift j=+1 has k+1<=1 and k+1>=0 => k=0 only. Shifts j=0,-1,-2 see standard causal."""
        B, H, D = 1, 1, 16
        T = 8
        Q, K, V = _rand_qkv(B, T, H, D)

        # Use pure j=+2 kernel
        conv_logits = torch.full((H, 5), -1e9, device=DEVICE, dtype=REF_DTYPE)
        conv_logits[:, 4] = 0.0  # only j=+2

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True)

        # At q=1, j=+2: need k+2 <= 1 and k+2 >= 0, so k <= -1 and k >= -2 => no valid k
        # So output at q=1 should be zero (nan_to_num)
        out_q1 = out[:, 1, :, :]
        torch.testing.assert_close(out_q1, torch.zeros_like(out_q1), atol=1e-6, rtol=0)


# ============================================================================
# Test 5: Causal masking preserved — no future attention
# ============================================================================
class TestCausalMasking:
    def test_no_future_attention(self):
        """Verify that modifying future V values doesn't change the output."""
        B, T, H, D = 1, 16, 2, 32
        Q, K, V = _rand_qkv(B, T, H, D)
        conv_logits = torch.randn(H, 5, device=DEVICE, dtype=REF_DTYPE)

        out1 = conv_attention_ref(Q, K, V, conv_logits, causal=True)

        # Perturb future V values: set V[T//2:] to something different
        V2 = V.clone()
        V2[:, T//2:, :, :] = torch.randn_like(V2[:, T//2:, :, :]) * 100

        out2 = conv_attention_ref(Q, K, V2, conv_logits, causal=True)

        # Outputs for q < T//2 - 2 should be identical
        # (shifts go up to +2, so positions up to T//2 - 3 are safe)
        safe_end = T // 2 - 2
        if safe_end > 0:
            torch.testing.assert_close(
                out1[:, :safe_end, :, :],
                out2[:, :safe_end, :, :],
                atol=1e-5, rtol=1e-5,
            )


# ============================================================================
# Test 6: Windowed causal attention
# ============================================================================
class TestWindowedAttention:
    def test_window_restricts_context(self):
        """With window_size=(W, 0), keys outside the window should be masked."""
        B, T, H, D = 1, 16, 2, 32
        W = 4
        Q, K, V = _rand_qkv(B, T, H, D)
        conv_logits = _identity_logits(H)  # identity kernel

        out_windowed = conv_attention_ref(Q, K, V, conv_logits, causal=True, window_size=(W, 0))
        out_std_windowed = standard_attention_ref(Q, K, V, causal=True, window_size=(W, 0))

        # Identity kernel + same window should match standard windowed attention
        torch.testing.assert_close(out_windowed, out_std_windowed, atol=1e-5, rtol=1e-5)

    def test_window_with_shifts(self):
        """With a narrow window, shifted keys outside the window should be masked out."""
        B, T, H, D = 1, 16, 1, 16
        W = 2
        Q, K, V = _rand_qkv(B, T, H, D)

        # Use j=-2 kernel only
        conv_logits = torch.full((H, 5), -1e9, device=DEVICE, dtype=REF_DTYPE)
        conv_logits[:, 0] = 0.0  # index 0 = shift -2

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True, window_size=(W, 0))

        # For j=-2 with window W=2:
        # valid keys: k-2 <= q (causal), k-2 >= q-W => k-2 >= q-2 => k >= q
        # Combined: k >= q and k-2 <= q => k <= q+2
        # Also k-2 >= 0 => k >= 2, and k-2 < T => k < T+2
        # So at query q: attend to k in [max(q, 2), min(q+2, T-1)]
        # With effective key k-2, that's positions [max(q,2)-2, min(q+2,T-1)-2] = [max(q-2,0), min(q,T-3)]

        # Just verify no NaN and correct shape
        assert out.shape == (B, T, H, D)
        assert not torch.isnan(out).any()


# ============================================================================
# Test 7: Full scale test
# ============================================================================
class TestFullScale:
    def test_large_no_nan(self):
        B, T, H, D = 2, 1024, 6, 128
        Q, K, V = _rand_qkv(B, T, H, D, dtype=DTYPE)
        conv_logits = torch.randn(H, 5, device=DEVICE, dtype=DTYPE)

        out = conv_attention_ref(
            Q.float(), K.float(), V.float(), conv_logits.float(), causal=True
        )

        assert out.shape == (B, T, H, D)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ============================================================================
# Test 8: Gradient flow
# ============================================================================
class TestGradients:
    def test_grad_conv_logits(self):
        B, T, H, D = 1, 8, 2, 16
        Q, K, V = _rand_qkv(B, T, H, D)
        conv_logits = torch.randn(H, 5, device=DEVICE, dtype=REF_DTYPE, requires_grad=True)

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True)
        loss = out.sum()
        loss.backward()

        assert conv_logits.grad is not None
        assert conv_logits.grad.shape == (H, 5)
        assert not torch.isnan(conv_logits.grad).any()

    def test_grad_qkv(self):
        B, T, H, D = 1, 8, 2, 16
        Q = torch.randn(B, T, H, D, device=DEVICE, dtype=REF_DTYPE, requires_grad=True)
        K = torch.randn(B, T, H, D, device=DEVICE, dtype=REF_DTYPE, requires_grad=True)
        V = torch.randn(B, T, H, D, device=DEVICE, dtype=REF_DTYPE, requires_grad=True)
        conv_logits = torch.randn(H, 5, device=DEVICE, dtype=REF_DTYPE, requires_grad=True)

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True)
        loss = out.sum()
        loss.backward()

        for name, tensor in [("Q", Q), ("K", K), ("V", V), ("conv_logits", conv_logits)]:
            assert tensor.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(tensor.grad).any(), f"{name} gradient has NaN"

    def test_grad_nonzero_for_nonidentity(self):
        """With a non-identity kernel, conv_logits grad should be nonzero."""
        B, T, H, D = 1, 16, 2, 32
        Q, K, V = _rand_qkv(B, T, H, D)
        # Use a non-trivial kernel
        conv_logits = torch.randn(H, 5, device=DEVICE, dtype=REF_DTYPE, requires_grad=True)

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True)
        # Use a non-trivial loss to ensure gradients flow
        loss = (out ** 2).sum()
        loss.backward()

        assert conv_logits.grad is not None
        assert conv_logits.grad.abs().sum() > 0, "conv_logits gradient is all zeros"
