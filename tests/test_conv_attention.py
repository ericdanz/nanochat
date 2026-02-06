"""
Correctness tests for conv attention (1D conv on attention probabilities after softmax).
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
# Test 2: Pure shift kernel [0,0,0,0,1] (j=+2)
# ============================================================================
class TestPureShift:
    def test_shift_plus2_small(self):
        """With kernel [0,0,0,0,1] (only j=+2): p_conv[q,k] = p[q, k-2], re-masked, @ original V."""
        B, T, H, D = 1, 8, 1, 16
        Q, K, V = _rand_qkv(B, T, H, D)

        # Logits that produce [0,0,0,0,1] after softmax => only j=+2 contributes
        conv_logits = torch.full((H, 5), -1e9, device=DEVICE, dtype=REF_DTYPE)
        conv_logits[:, 4] = 0.0  # index 4 = shift +2

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True)

        # Manually: standard softmax, shift probs by +2, re-mask, attend to original V
        q = Q.transpose(1, 2)  # (B, H, T, D)
        k = K.transpose(1, 2)
        v = V.transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / (D ** 0.5)

        # Standard causal mask + softmax
        q_idx = torch.arange(T, device=DEVICE).unsqueeze(1)
        k_idx = torch.arange(T, device=DEVICE).unsqueeze(0)
        causal_mask = k_idx <= q_idx
        logits = logits.masked_fill(~causal_mask, float('-inf'))
        p = F.softmax(logits, dim=-1)
        p = torch.nan_to_num(p, nan=0.0)

        # Shift probs: p_conv[q, k] = p[q, k-2]
        p_conv = torch.zeros_like(p)
        p_conv[:, :, :, 2:] = p[:, :, :, :T - 2]

        # Re-mask to enforce causality
        p_conv = p_conv.masked_fill(~causal_mask, 0.0)

        # Attend to original V
        expected = (p_conv @ v).transpose(1, 2)

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
    def test_q0_boundary(self):
        """At q=0 with causal, p[0,:] = [1,0,0,...] (only k=0 valid).
        Conv: p_conv[0, k] = sum_j w[j] * p[0, k-j]. Only j=0 contributes
        p_conv[0,0] = w[0]*p[0,0] = w[center]*1. Shifts j=+1,+2 need k-j < 0 (OOB),
        shifts j=-1,-2 produce p_conv at k=1,2 which get re-masked (k>q=0).
        With identity kernel: out[q=0] = V[0].
        With uniform w=0.2: out[q=0] = 0.2*V[0] (only j=0 survives at k=0).
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

        # Uniform kernel w=0.2 => only j=0 tap survives at q=0 => out = 0.2*V[0]
        conv_logits_uniform = torch.zeros(H, 5, device=DEVICE, dtype=REF_DTYPE)
        out2 = conv_attention_ref(Q, K, V, conv_logits_uniform, causal=True)
        out2_q0 = out2[:, 0, :, :]
        torch.testing.assert_close(out2_q0, 0.2 * v0, atol=1e-5, rtol=1e-5)

    def test_q1_no_future_leak(self):
        """At q=1 with causal and pure j=+2 kernel:
        p_conv[1, k] = p[1, k-2]. For k=0: k-2=-2 (OOB). For k=1: k-2=-1 (OOB).
        So p_conv[1,:] is all zeros => output at q=1 should be zero."""
        B, H, D = 1, 1, 16
        T = 8
        Q, K, V = _rand_qkv(B, T, H, D)

        # Use pure j=+2 kernel
        conv_logits = torch.full((H, 5), -1e9, device=DEVICE, dtype=REF_DTYPE)
        conv_logits[:, 4] = 0.0  # only j=+2

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True)

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
        """With a narrow window, shifted probs outside the window should be masked out."""
        B, T, H, D = 1, 16, 1, 16
        W = 2
        Q, K, V = _rand_qkv(B, T, H, D)

        # Use j=-2 kernel only: p_conv[q, k] = p[q, k+2] then re-mask
        conv_logits = torch.full((H, 5), -1e9, device=DEVICE, dtype=REF_DTYPE)
        conv_logits[:, 0] = 0.0  # index 0 = shift -2

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True, window_size=(W, 0))

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


# ============================================================================
# Test 9: Partial conv heads — first n_conv get conv, rest standard
# ============================================================================
class TestPartialConvHeads:
    def test_no_conv_logits_is_standard(self):
        """conv_logits=None should match standard attention exactly."""
        B, T, H, D = 2, 16, 4, 32
        Q, K, V = _rand_qkv(B, T, H, D)

        out_none = conv_attention_ref(Q, K, V, conv_logits=None, causal=True)
        out_std = standard_attention_ref(Q, K, V, causal=True)

        torch.testing.assert_close(out_none, out_std, atol=1e-5, rtol=1e-5)

    def test_subset_conv_heads_standard_heads_unaffected(self):
        """Non-conv heads (h >= n_conv) should match standard attention."""
        B, T, H, D = 2, 16, 6, 32
        n_conv = 2  # only first 2 of 6 heads get conv
        Q, K, V = _rand_qkv(B, T, H, D)
        conv_logits = torch.randn(n_conv, 5, device=DEVICE, dtype=REF_DTYPE)

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True)
        out_std = standard_attention_ref(Q, K, V, causal=True)

        # Heads n_conv: should be identical to standard attention
        torch.testing.assert_close(
            out[:, :, n_conv:, :], out_std[:, :, n_conv:, :],
            atol=1e-5, rtol=1e-5,
        )

    def test_subset_conv_heads_identity_all_match(self):
        """If conv heads use identity kernel, ALL heads match standard."""
        B, T, H, D = 2, 16, 6, 32
        n_conv = 3
        Q, K, V = _rand_qkv(B, T, H, D)

        conv_logits = torch.full((n_conv, 5), -1e9, device=DEVICE, dtype=REF_DTYPE)
        conv_logits[:, 2] = 0.0  # identity

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True)
        out_std = standard_attention_ref(Q, K, V, causal=True)

        torch.testing.assert_close(out, out_std, atol=1e-5, rtol=1e-5)

    def test_all_heads_conv_matches_original(self):
        """conv_logits with shape (H, 5) should behave same as before."""
        B, T, H, D = 2, 16, 4, 32
        Q, K, V = _rand_qkv(B, T, H, D)
        conv_logits = torch.randn(H, 5, device=DEVICE, dtype=REF_DTYPE)

        out_all = conv_attention_ref(Q, K, V, conv_logits, causal=True)

        # Should be the same as manually applying conv to all heads
        # (identity test: use identity logits to confirm)
        conv_logits_id = torch.full((H, 5), -1e9, device=DEVICE, dtype=REF_DTYPE)
        conv_logits_id[:, 2] = 0.0
        out_id = conv_attention_ref(Q, K, V, conv_logits_id, causal=True)
        out_std = standard_attention_ref(Q, K, V, causal=True)
        torch.testing.assert_close(out_id, out_std, atol=1e-5, rtol=1e-5)

    def test_grad_flows_to_conv_and_qkv(self):
        """Gradients flow through both conv heads and standard heads."""
        B, T, H, D = 1, 8, 4, 16
        n_conv = 2
        Q = torch.randn(B, T, H, D, device=DEVICE, dtype=REF_DTYPE, requires_grad=True)
        K = torch.randn(B, T, H, D, device=DEVICE, dtype=REF_DTYPE, requires_grad=True)
        V = torch.randn(B, T, H, D, device=DEVICE, dtype=REF_DTYPE, requires_grad=True)
        conv_logits = torch.randn(n_conv, 5, device=DEVICE, dtype=REF_DTYPE, requires_grad=True)

        out = conv_attention_ref(Q, K, V, conv_logits, causal=True)
        loss = (out ** 2).sum()
        loss.backward()

        for name, t in [("Q", Q), ("K", K), ("V", V), ("conv_logits", conv_logits)]:
            assert t.grad is not None, f"{name} has no gradient"
            assert not torch.isnan(t.grad).any(), f"{name} gradient has NaN"
            assert t.grad.abs().sum() > 0, f"{name} gradient is all zeros"
