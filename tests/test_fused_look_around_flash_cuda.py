"""
Tests for the CUDA fused look-around flash attention kernel.

These tests verify:
1. Forward correctness: Output matches reference implementation
2. Backward correctness: Gradients match numerical differentiation
3. Causality preservation: No gradient leakage to future positions
4. Gradient magnitude: Q/K gradients are NOT 100-700x off (Triton bug is fixed)
"""

import pytest
import torch
import torch.nn.functional as F
import math

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)

# Try to import CUDA kernel
try:
    from nanochat.cuda_kernels import (
        fused_look_around_flash_attention_cuda,
        is_cuda_kernel_available,
    )
    CUDA_KERNEL_AVAILABLE = is_cuda_kernel_available()
except ImportError:
    CUDA_KERNEL_AVAILABLE = False

# Reference implementation in pure PyTorch
def fused_look_around_flash_attention_reference(
    q, k, v, proj_logits, causal=True, window_left=-1
):
    """
    Reference implementation of fused look-around flash attention in PyTorch.

    The convolution formula: P_conv[j] = w0*P[j+2] + w1*P[j+1] + w2*P[j] + w3*P[j-1] + w4*P[j-2]

    IMPORTANT: The CUDA kernel uses a SHARED softmax normalization across all 5 shifted scores.
    This means all exp(score) values share the same max and denominator. The convolution
    weights are applied to the unnormalized exp values, and normalization happens on the
    convolved result.
    """
    B, H, T_q, D = q.shape
    T_k = k.shape[2]

    scale = 1.0 / math.sqrt(D)
    # (B, H, T_q, T_k)
    scores = torch.matmul(q.float(), k.transpose(-2, -1).float()) * scale

    # Softmax projection weights: [w0, w1, w2, w3, w4] for shifts [+2, +1, 0, -1, -2]
    w = F.softmax(proj_logits.float(), dim=-1)  # (H, 5)

    # For each output V position j, we need scores at K positions j+2, j+1, j, j-1, j-2
    # We'll compute the convolved attention weight using a shared softmax denominator

    # Create shifted score tensors (pad with -inf for out-of-bounds)
    def get_shifted_score(shift):
        """Get scores shifted by `shift` positions. P_conv[j] uses score at K position j+shift."""
        if shift == 0:
            return scores
        elif shift > 0:
            # For P_conv[j], we want score for K[j+shift]
            # This means shifting the score array left
            return torch.cat([scores[:, :, :, shift:],
                            torch.full((B, H, T_q, shift), float('-inf'), device=scores.device)], dim=-1)
        else:
            # For P_conv[j], we want score for K[j+shift] = K[j-|shift|]
            # This means shifting the score array right
            shift = abs(shift)
            return torch.cat([torch.full((B, H, T_q, shift), float('-inf'), device=scores.device),
                            scores[:, :, :, :-shift]], dim=-1)

    # Get all 5 shifted score arrays
    s_p2 = get_shifted_score(2)   # For P_conv[j], this is score for K[j+2]
    s_p1 = get_shifted_score(1)   # For P_conv[j], this is score for K[j+1]
    s_0  = get_shifted_score(0)   # For P_conv[j], this is score for K[j]
    s_m1 = get_shifted_score(-1)  # For P_conv[j], this is score for K[j-1]
    s_m2 = get_shifted_score(-2)  # For P_conv[j], this is score for K[j-2]

    # Apply causal mask to each shifted score array
    if causal:
        # For position (q, j), the K position used is j+shift
        # Causal: mask if k_pos > q_pos, i.e., j+shift > q
        for shift, s in [(2, s_p2), (1, s_p1), (0, s_0), (-1, s_m1), (-2, s_m2)]:
            q_idx = torch.arange(T_q, device=q.device).view(1, 1, T_q, 1)
            k_idx = torch.arange(T_k, device=q.device).view(1, 1, 1, T_k) + shift
            causal_mask = k_idx > q_idx
            s.masked_fill_(causal_mask, float('-inf'))

    # Stack all shifted scores: (B, H, T_q, T_k, 5)
    all_scores = torch.stack([s_p2, s_p1, s_0, s_m1, s_m2], dim=-1)

    # Compute GLOBAL max across all 5 shifts for numerical stability
    max_score = all_scores.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]  # (B, H, T_q, 1, 1)
    max_score = torch.clamp(max_score, min=-1e30)  # Avoid -inf

    # Compute exp(score - max) for each shift
    exp_scores = torch.exp(all_scores - max_score)  # (B, H, T_q, T_k, 5)

    # Apply convolution weights: w0*exp_p2 + w1*exp_p1 + w2*exp_0 + w3*exp_m1 + w4*exp_m2
    w_expanded = w.view(1, H, 1, 1, 5)  # (1, H, 1, 1, 5)
    p_conv_unnorm = (exp_scores * w_expanded).sum(dim=-1)  # (B, H, T_q, T_k)

    # Re-apply causal mask on V positions (convolution can bring in valid attention for wrong V)
    if causal:
        v_mask = torch.triu(torch.ones(T_q, T_k, device=q.device), diagonal=1).bool()
        p_conv_unnorm = p_conv_unnorm.masked_fill(v_mask.view(1, 1, T_q, T_k), 0.0)

    # Normalize
    p_conv = p_conv_unnorm / (p_conv_unnorm.sum(dim=-1, keepdim=True) + 1e-9)

    out = torch.matmul(p_conv.to(v.dtype), v)
    return out


def skip_if_kernel_not_built():
    """Skip test if CUDA kernel is not built."""
    if not CUDA_KERNEL_AVAILABLE:
        pytest.skip("CUDA kernel not built. Run: cd nanochat/cuda_kernels && pip install -e .")


class TestForwardCorrectness:
    """Tests for forward pass correctness."""

    def test_basic_forward_shape(self):
        """Test that output has correct shape."""
        skip_if_kernel_not_built()

        B, H, T, D = 2, 4, 128, 64
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
        proj_logits = torch.randn(H, 5, device="cuda", dtype=torch.float32)

        out = fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal=True)

        assert out.shape == (B, H, T, D)
        assert out.dtype == torch.bfloat16  # Kernel outputs bfloat16
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_matches_reference(self):
        """Test that CUDA kernel matches reference implementation."""
        skip_if_kernel_not_built()

        torch.manual_seed(42)
        B, H, T, D = 1, 2, 64, 32
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        proj_logits = torch.randn(H, 5, device="cuda", dtype=torch.float32)

        # CUDA kernel (converts to bf16 internally)
        out_cuda = fused_look_around_flash_attention_cuda(
            q, k, v, proj_logits, causal=True
        )

        # Reference implementation
        out_ref = fused_look_around_flash_attention_reference(
            q, k, v, proj_logits, causal=True
        )

        # Compare with tolerance for bf16 precision
        torch.testing.assert_close(
            out_cuda.float(), out_ref.float(), rtol=0.02, atol=0.01
        )

    def test_non_causal(self):
        """Test non-causal mode."""
        skip_if_kernel_not_built()

        B, H, T, D = 1, 2, 64, 32
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        proj_logits = torch.randn(H, 5, device="cuda", dtype=torch.float32)

        out = fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal=False)

        assert out.shape == (B, H, T, D)
        assert not torch.isnan(out).any()

    def test_various_sequence_lengths(self):
        """Test with sequence lengths that don't align with block size."""
        skip_if_kernel_not_built()

        for T in [37, 65, 100, 129, 256]:
            B, H, D = 1, 2, 32
            q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
            k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
            v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
            proj_logits = torch.randn(H, 5, device="cuda", dtype=torch.float32)

            out = fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal=True)

            assert out.shape == (B, H, T, D), f"Shape mismatch for T={T}"
            assert not torch.isnan(out).any(), f"NaN for T={T}"

    def test_identity_projection(self):
        """Test that identity projection matches standard attention."""
        skip_if_kernel_not_built()

        torch.manual_seed(42)
        B, H, T, D = 1, 2, 64, 32
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)

        # Identity projection: only center weight is 1
        proj_logits = torch.tensor(
            [[-10.0, -10.0, 0.0, -10.0, -10.0]] * H,
            device="cuda", dtype=torch.float32
        )

        out_cuda = fused_look_around_flash_attention_cuda(
            q, k, v, proj_logits, causal=True
        )

        # Standard causal attention
        scale = 1.0 / math.sqrt(D)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        mask = torch.triu(torch.ones(T, T, device="cuda"), diagonal=1).bool()
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out_std = torch.matmul(attn, v)

        # Should be close with identity projection
        torch.testing.assert_close(
            out_cuda.float(), out_std.float(), rtol=0.05, atol=0.05
        )


class TestBackwardCorrectness:
    """Tests for backward pass correctness - the main focus since Triton backward was buggy."""

    def test_gradients_exist(self):
        """Test that gradients flow through the kernel."""
        skip_if_kernel_not_built()

        B, H, T, D = 1, 2, 32, 32
        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        proj_logits = torch.randn(H, 5, device="cuda", dtype=torch.float32, requires_grad=True)

        out = fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal=True)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None, "q.grad is None"
        assert k.grad is not None, "k.grad is None"
        assert v.grad is not None, "v.grad is None"
        assert proj_logits.grad is not None, "proj_logits.grad is None"

        assert not torch.isnan(q.grad).any(), "q.grad has NaN"
        assert not torch.isnan(k.grad).any(), "k.grad has NaN"
        assert not torch.isnan(v.grad).any(), "v.grad has NaN"
        assert not torch.isnan(proj_logits.grad).any(), "proj_logits.grad has NaN"

    def test_gradient_magnitude_not_exploding(self):
        """
        Test that Q/K gradients are NOT 100-700x off like in Triton.

        This is the key test that verifies the backward bugs are fixed.
        """
        skip_if_kernel_not_built()

        torch.manual_seed(42)
        B, H, T, D = 1, 2, 32, 32

        # CUDA kernel
        q_cuda = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        k_cuda = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        v_cuda = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        proj_cuda = torch.randn(H, 5, device="cuda", dtype=torch.float32, requires_grad=True)

        out_cuda = fused_look_around_flash_attention_cuda(
            q_cuda, k_cuda, v_cuda, proj_cuda, causal=True
        )
        loss_cuda = out_cuda.float().sum()
        loss_cuda.backward()

        # Reference implementation
        q_ref = q_cuda.detach().clone().requires_grad_(True)
        k_ref = k_cuda.detach().clone().requires_grad_(True)
        v_ref = v_cuda.detach().clone().requires_grad_(True)
        proj_ref = proj_cuda.detach().clone().requires_grad_(True)

        out_ref = fused_look_around_flash_attention_reference(
            q_ref, k_ref, v_ref, proj_ref, causal=True
        )
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # Gradient ratios should be close to 1, not 100-700x
        v_ratio = v_cuda.grad.abs().max() / (v_ref.grad.abs().max() + 1e-8)
        k_ratio = k_cuda.grad.abs().max() / (k_ref.grad.abs().max() + 1e-8)
        q_ratio = q_cuda.grad.abs().max() / (q_ref.grad.abs().max() + 1e-8)

        print(f"\nGradient magnitude ratios (CUDA / Reference):")
        print(f"  V.grad ratio: {v_ratio.item():.2f}x")
        print(f"  K.grad ratio: {k_ratio.item():.2f}x")
        print(f"  Q.grad ratio: {q_ratio.item():.2f}x")

        # Gradients should be within 10x, not 100-700x
        assert v_ratio < 10, f"V gradient ratio {v_ratio:.1f}x is too large"
        assert k_ratio < 10, f"K gradient ratio {k_ratio:.1f}x is too large"
        assert q_ratio < 10, f"Q gradient ratio {q_ratio:.1f}x is too large"

    def test_gradients_match_reference(self):
        """Test that gradients match reference implementation closely."""
        skip_if_kernel_not_built()

        torch.manual_seed(42)
        B, H, T, D = 1, 2, 32, 32

        # CUDA kernel
        q_cuda = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        k_cuda = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        v_cuda = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        proj_cuda = torch.randn(H, 5, device="cuda", dtype=torch.float32, requires_grad=True)

        out_cuda = fused_look_around_flash_attention_cuda(
            q_cuda, k_cuda, v_cuda, proj_cuda, causal=True
        )
        loss_cuda = out_cuda.float().sum()
        loss_cuda.backward()

        # Reference implementation
        q_ref = q_cuda.detach().clone().requires_grad_(True)
        k_ref = k_cuda.detach().clone().requires_grad_(True)
        v_ref = v_cuda.detach().clone().requires_grad_(True)
        proj_ref = proj_cuda.detach().clone().requires_grad_(True)

        out_ref = fused_look_around_flash_attention_reference(
            q_ref, k_ref, v_ref, proj_ref, causal=True
        )
        loss_ref = out_ref.sum()
        loss_ref.backward()

        # Compare gradients (with tolerance for bf16 precision loss)
        rtol, atol = 0.1, 0.05

        v_diff = (v_cuda.grad - v_ref.grad).abs().max().item()
        v_scale = max(v_cuda.grad.abs().max().item(), v_ref.grad.abs().max().item(), 1e-6)
        assert v_diff < atol + rtol * v_scale, f"V gradient mismatch: diff={v_diff:.4f}"

        k_diff = (k_cuda.grad - k_ref.grad).abs().max().item()
        k_scale = max(k_cuda.grad.abs().max().item(), k_ref.grad.abs().max().item(), 1e-6)
        assert k_diff < atol + rtol * k_scale, f"K gradient mismatch: diff={k_diff:.4f}"

        q_diff = (q_cuda.grad - q_ref.grad).abs().max().item()
        q_scale = max(q_cuda.grad.abs().max().item(), q_ref.grad.abs().max().item(), 1e-6)
        assert q_diff < atol + rtol * q_scale, f"Q gradient mismatch: diff={q_diff:.4f}"


class TestCausality:
    """Tests for causality preservation - ensures no gradient leakage to future positions."""

    def test_forward_perturbation(self):
        """Test that perturbing future V values doesn't change past outputs."""
        skip_if_kernel_not_built()

        torch.manual_seed(42)
        B, H, T, D = 1, 2, 32, 32

        for test_pos in [0, T // 4, T // 2, T - 2]:
            q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
            k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
            v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
            proj_logits = torch.randn(H, 5, device="cuda", dtype=torch.float32)

            out_original = fused_look_around_flash_attention_cuda(
                q, k, v, proj_logits, causal=True
            )

            # Perturb V at positions > test_pos
            v_perturbed = v.clone()
            v_perturbed[:, :, test_pos + 1:, :] += 1000.0

            out_perturbed = fused_look_around_flash_attention_cuda(
                q, k, v_perturbed, proj_logits, causal=True
            )

            # Output at positions <= test_pos should be identical
            diff = (out_original[:, :, :test_pos + 1, :] -
                    out_perturbed[:, :, :test_pos + 1, :]).abs().max().item()
            assert diff < 1e-5, (
                f"Forward causality violation at test_pos={test_pos}: "
                f"perturbing V[>{test_pos}] changed output[<={test_pos}] by {diff:.6e}"
            )

    def test_no_gradient_leakage(self):
        """Test that gradients don't flow from future to past positions."""
        skip_if_kernel_not_built()

        torch.manual_seed(42)
        B, H, T, D = 1, 2, 32, 32

        test_positions = [0, T // 4, T // 2, 3 * T // 4]

        for loss_pos in test_positions:
            q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
            k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
            v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
            proj_logits = torch.randn(H, 5, device="cuda", dtype=torch.float32, requires_grad=True)

            out = fused_look_around_flash_attention_cuda(
                q, k, v, proj_logits, causal=True
            )

            # Loss only from position loss_pos
            loss = out[:, :, loss_pos, :].sum()
            loss.backward()

            # V gradients at positions > loss_pos should be zero
            v_grad_future = v.grad[:, :, loss_pos + 1:, :].abs().max().item()
            assert v_grad_future < 1e-5, (
                f"V gradient leakage at loss_pos={loss_pos}: {v_grad_future:.6e}"
            )

            # K gradients at positions > loss_pos should be zero
            k_grad_future = k.grad[:, :, loss_pos + 1:, :].abs().max().item()
            assert k_grad_future < 1e-5, (
                f"K gradient leakage at loss_pos={loss_pos}: {k_grad_future:.6e}"
            )

    def test_extreme_future_weights(self):
        """Test causality with projection weights favoring future positions."""
        skip_if_kernel_not_built()

        torch.manual_seed(42)
        B, H, T, D = 1, 2, 64, 32

        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)

        # Extreme weights favoring +2, +1 shifts
        proj_logits = torch.tensor(
            [[10.0, 5.0, -10.0, -10.0, -10.0]] * H,
            device="cuda", dtype=torch.float32
        )

        out = fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal=True)

        assert not torch.isnan(out).any(), "NaN with future-weighted projection"
        assert not torch.isinf(out).any(), "Inf with future-weighted projection"

        # Compute loss from early position
        loss_pos = T // 4
        loss = out[:, :, loss_pos, :].sum()
        loss.backward()

        # Even with extreme future weights, no gradient leakage
        v_grad_future = v.grad[:, :, loss_pos + 1:, :].abs().max().item()
        assert v_grad_future < 1e-5, f"Gradient leakage with future weights: {v_grad_future:.6e}"


class TestNumericalGradient:
    """Tests using numerical differentiation to verify gradients."""

    def test_v_gradient_numerical(self):
        """Verify V gradient using numerical differentiation."""
        skip_if_kernel_not_built()

        torch.manual_seed(42)
        B, H, T, D = 1, 1, 32, 32
        eps = 1e-3

        def func(v):
            out = fused_look_around_flash_attention_cuda(
                q, k, v, proj_logits, causal=True
            )
            return out.float().sum()

        q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        proj_logits = torch.randn(H, 5, device="cuda", dtype=torch.float32)

        # Analytical gradient
        out = func(v)
        out.backward()
        grad_analytical = v.grad.clone()

        # Numerical gradient for a few elements
        for idx in [(0, 0, T // 4, D // 2), (0, 0, T // 2, 0)]:
            v_plus = v.detach().clone()
            v_plus[idx] += eps
            v_minus = v.detach().clone()
            v_minus[idx] -= eps

            grad_numerical = (func(v_plus) - func(v_minus)) / (2 * eps)

            diff = abs(grad_analytical[idx].item() - grad_numerical.item())
            scale = max(abs(grad_analytical[idx].item()), abs(grad_numerical.item()), 1e-6)

            assert diff < 0.1 + 0.2 * scale, (
                f"V gradient mismatch at {idx}: "
                f"analytical={grad_analytical[idx].item():.4f}, "
                f"numerical={grad_numerical.item():.4f}"
            )


class TestGQA:
    """Tests for Grouped Query Attention (GQA) support."""

    def test_gqa_forward_shape(self):
        """Test that GQA output has correct shape."""
        skip_if_kernel_not_built()

        B, T, D = 2, 64, 32
        n_q_heads = 6
        n_kv_heads = 2

        q = torch.randn(B, n_q_heads, T, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32)
        proj_logits = torch.randn(n_kv_heads, 5, device="cuda", dtype=torch.float32)

        out = fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal=True)

        assert out.shape == (B, n_q_heads, T, D), f"Expected shape {(B, n_q_heads, T, D)}, got {out.shape}"
        assert out.dtype == torch.bfloat16
        assert not torch.isnan(out).any()

    def test_gqa_gradient_shapes(self):
        """Test that GQA gradients have correct shapes."""
        skip_if_kernel_not_built()

        B, T, D = 2, 64, 32
        n_q_heads = 6
        n_kv_heads = 2

        q = torch.randn(B, n_q_heads, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        proj_logits = torch.randn(n_kv_heads, 5, device="cuda", dtype=torch.float32, requires_grad=True)

        out = fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal=True)
        loss = out.float().sum()
        loss.backward()

        assert q.grad.shape == (B, n_q_heads, T, D), f"q.grad shape mismatch: {q.grad.shape}"
        assert k.grad.shape == (B, n_kv_heads, T, D), f"k.grad shape mismatch: {k.grad.shape}"
        assert v.grad.shape == (B, n_kv_heads, T, D), f"v.grad shape mismatch: {v.grad.shape}"
        assert proj_logits.grad.shape == (n_kv_heads, 5), f"proj_logits.grad shape mismatch: {proj_logits.grad.shape}"

        assert not torch.isnan(q.grad).any(), "q.grad has NaN"
        assert not torch.isnan(k.grad).any(), "k.grad has NaN"
        assert not torch.isnan(v.grad).any(), "v.grad has NaN"
        assert not torch.isnan(proj_logits.grad).any(), "proj_logits.grad has NaN"

    def test_gqa_matches_expanded(self):
        """Test GQA matches manually expanded K/V."""
        skip_if_kernel_not_built()

        torch.manual_seed(42)
        B, T, D = 1, 32, 32
        n_q_heads = 6
        n_kv_heads = 2

        q = torch.randn(B, n_q_heads, T, D, device="cuda", dtype=torch.float32)
        k_base = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32)
        v_base = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32)
        proj_logits_base = torch.randn(n_kv_heads, 5, device="cuda", dtype=torch.float32)

        # GQA result
        out_gqa = fused_look_around_flash_attention_cuda(
            q, k_base, v_base, proj_logits_base, causal=True
        )

        # Manually expanded K/V (repeat each KV head for corresponding Q heads)
        heads_per_kv = n_q_heads // n_kv_heads
        k_expanded = k_base.repeat_interleave(heads_per_kv, dim=1)
        v_expanded = v_base.repeat_interleave(heads_per_kv, dim=1)
        proj_logits_expanded = proj_logits_base.repeat_interleave(heads_per_kv, dim=0)

        out_expanded = fused_look_around_flash_attention_cuda(
            q, k_expanded, v_expanded, proj_logits_expanded, causal=True
        )

        # Results should be identical
        torch.testing.assert_close(
            out_gqa.float(), out_expanded.float(), rtol=1e-4, atol=1e-4
        )

    def test_gqa_various_ratios(self):
        """Test GQA with different head ratios."""
        skip_if_kernel_not_built()

        B, T, D = 1, 32, 32

        # Test various GQA ratios
        test_configs = [
            (4, 2),   # 2:1 ratio
            (6, 2),   # 3:1 ratio
            (6, 3),   # 2:1 ratio
            (8, 2),   # 4:1 ratio
            (12, 3),  # 4:1 ratio
        ]

        for n_q_heads, n_kv_heads in test_configs:
            q = torch.randn(B, n_q_heads, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
            k = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
            v = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
            proj_logits = torch.randn(n_kv_heads, 5, device="cuda", dtype=torch.float32, requires_grad=True)

            out = fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal=True)
            loss = out.float().sum()
            loss.backward()

            assert out.shape == (B, n_q_heads, T, D), f"Shape mismatch for ratio {n_q_heads}:{n_kv_heads}"
            assert not torch.isnan(out).any(), f"NaN for ratio {n_q_heads}:{n_kv_heads}"
            assert not torch.isnan(q.grad).any(), f"NaN in q.grad for ratio {n_q_heads}:{n_kv_heads}"
            assert not torch.isnan(k.grad).any(), f"NaN in k.grad for ratio {n_q_heads}:{n_kv_heads}"

    def test_gqa_invalid_ratio(self):
        """Test that invalid GQA ratios raise errors."""
        skip_if_kernel_not_built()

        B, T, D = 1, 32, 32
        n_q_heads = 5  # Not divisible by 2
        n_kv_heads = 2

        q = torch.randn(B, n_q_heads, T, D, device="cuda", dtype=torch.float32)
        k = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32)
        v = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32)
        proj_logits = torch.randn(n_kv_heads, 5, device="cuda", dtype=torch.float32)

        with pytest.raises(ValueError, match="divisible"):
            fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal=True)

    def test_gqa_causality(self):
        """Test that GQA preserves causal masking."""
        skip_if_kernel_not_built()

        torch.manual_seed(42)
        B, T, D = 1, 32, 32
        n_q_heads = 6
        n_kv_heads = 2

        for test_pos in [0, T // 4, T // 2]:
            q = torch.randn(B, n_q_heads, T, D, device="cuda", dtype=torch.float32)
            k = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32)
            v = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
            proj_logits = torch.randn(n_kv_heads, 5, device="cuda", dtype=torch.float32)

            out = fused_look_around_flash_attention_cuda(q, k, v, proj_logits, causal=True)

            # Loss from position test_pos
            loss = out[:, :, test_pos, :].sum()
            loss.backward()

            # V gradients at positions > test_pos should be zero
            v_grad_future = v.grad[:, :, test_pos + 1:, :].abs().max().item()
            assert v_grad_future < 1e-5, (
                f"V gradient leakage at test_pos={test_pos} with GQA: {v_grad_future:.6e}"
            )

    def test_gqa_with_window(self):
        """Test GQA with sliding window attention."""
        skip_if_kernel_not_built()

        B, T, D = 2, 64, 32
        n_q_heads = 6
        n_kv_heads = 2
        window_left = 16

        q = torch.randn(B, n_q_heads, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, n_kv_heads, T, D, device="cuda", dtype=torch.float32, requires_grad=True)
        proj_logits = torch.randn(n_kv_heads, 5, device="cuda", dtype=torch.float32, requires_grad=True)

        out = fused_look_around_flash_attention_cuda(
            q, k, v, proj_logits, causal=True, window_left=window_left
        )

        assert out.shape == (B, n_q_heads, T, D)
        assert not torch.isnan(out).any()

        loss = out.float().sum()
        loss.backward()

        assert not torch.isnan(q.grad).any()
        assert not torch.isnan(k.grad).any()
        assert not torch.isnan(v.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
