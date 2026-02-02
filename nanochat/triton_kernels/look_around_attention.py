"""
Triton kernel for look-around attention convolution.

This module provides a Triton-accelerated implementation of the look_around_conv function
that applies a 5-element 1D convolution to post-softmax attention weights.

The convolution formula is:
    new_attn[y] = attn[y+2]*proj[0] + attn[y+1]*proj[1] + attn[y]*proj[2] + attn[y-1]*proj[3] + attn[y-2]*proj[4]

After convolution, a causal mask is applied and the result is renormalized to sum to 1.
"""

import torch
import torch.nn.functional as F

# Try to import triton, fall back gracefully if not available
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def look_around_conv_pytorch(
    attn_weights: torch.Tensor, proj_logits: torch.Tensor
) -> torch.Tensor:
    """
    PyTorch reference implementation of look-around convolution.

    Args:
        attn_weights: (B, H, T_q, T_k) post-softmax attention weights
        proj_logits: (H, 5) learnable projection logits per head

    Returns:
        (B, H, T_q, T_k) convolved attention weights, summing to 1
    """
    B, H, T_q, T_k = attn_weights.shape

    # Compute softmax projection
    proj_5 = F.softmax(proj_logits, dim=-1)  # (H, 5)

    # Flip for convolution kernel
    kernel = proj_5.flip(-1).unsqueeze(1)  # (H, 1, 5)

    # Reshape for grouped 1D convolution: (B, H, T_q, T_k) -> (B * T_q, H, T_k)
    attn_flat = attn_weights.permute(0, 2, 1, 3).reshape(B * T_q, H, T_k)

    # Apply grouped 1D convolution with padding=2
    attn_conv = F.conv1d(attn_flat, kernel, padding=2, groups=H)

    # Reshape back to (B, H, T_q, T_k)
    attn_conv = attn_conv.reshape(B, T_q, H, T_k).permute(0, 2, 1, 3)

    # Zero out upper triangle (causal mask)
    row_idx = torch.arange(T_q, device=attn_weights.device).view(-1, 1)
    col_idx = torch.arange(T_k, device=attn_weights.device).view(1, -1)
    causal_mask = col_idx > row_idx
    attn_conv = attn_conv.masked_fill(causal_mask, 0.0)

    # Renormalize to sum to 1
    attn_conv = attn_conv / attn_conv.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    return attn_conv


if TRITON_AVAILABLE:
    @triton.jit
    def _look_around_conv_fwd_kernel(
        attn_ptr, proj_ptr, out_ptr,
        B, H, T_q, T_k,
        stride_ab, stride_ah, stride_aq, stride_ak,
        stride_pb, stride_p5,
        stride_ob, stride_oh, stride_oq, stride_ok,
        BLOCK_K: tl.constexpr,
    ):
        """
        Forward kernel for look-around convolution.

        Each program handles one (batch, head, query) combination.
        Processes key positions in blocks to apply convolution, causal mask, and renormalization.
        """
        # Program IDs
        pid_bq = tl.program_id(0)  # batch * query index
        pid_h = tl.program_id(1)   # head index

        pid_b = pid_bq // T_q
        pid_q = pid_bq % T_q

        # Load the 5-element projection for this head (already softmaxed)
        proj_base = proj_ptr + pid_h * stride_pb
        p0 = tl.load(proj_base + 0 * stride_p5)  # K-2
        p1 = tl.load(proj_base + 1 * stride_p5)  # K-1
        p2 = tl.load(proj_base + 2 * stride_p5)  # K (center)
        p3 = tl.load(proj_base + 3 * stride_p5)  # K+1
        p4 = tl.load(proj_base + 4 * stride_p5)  # K+2

        # Base pointer for this (batch, head, query) row in attention
        attn_base = attn_ptr + pid_b * stride_ab + pid_h * stride_ah + pid_q * stride_aq
        out_base = out_ptr + pid_b * stride_ob + pid_h * stride_oh + pid_q * stride_oq

        # First pass: compute convolved values and accumulate sum for normalization
        # Process in blocks of BLOCK_K
        # Initialize accumulator with explicit float32 type
        acc_sum: tl.float32 = 0.0

        for block_start in range(0, T_k, BLOCK_K):
            k_offsets = block_start + tl.arange(0, BLOCK_K)
            mask = k_offsets < T_k

            # Load attention values for convolution
            # For position k, we need: attn[k-2], attn[k-1], attn[k], attn[k+1], attn[k+2]
            # new_attn[k] = attn[k+2]*p0 + attn[k+1]*p1 + attn[k]*p2 + attn[k-1]*p3 + attn[k-2]*p4

            # Load with boundary handling (out of bounds -> 0)
            # Cast to float32 for consistent accumulation
            a_km2 = tl.load(attn_base + (k_offsets - 2) * stride_ak,
                           mask=mask & (k_offsets >= 2), other=0.0).to(tl.float32)
            a_km1 = tl.load(attn_base + (k_offsets - 1) * stride_ak,
                           mask=mask & (k_offsets >= 1), other=0.0).to(tl.float32)
            a_k = tl.load(attn_base + k_offsets * stride_ak, mask=mask, other=0.0).to(tl.float32)
            a_kp1 = tl.load(attn_base + (k_offsets + 1) * stride_ak,
                           mask=mask & (k_offsets + 1 < T_k), other=0.0).to(tl.float32)
            a_kp2 = tl.load(attn_base + (k_offsets + 2) * stride_ak,
                           mask=mask & (k_offsets + 2 < T_k), other=0.0).to(tl.float32)

            # Compute convolution
            conv_val = a_kp2 * p0 + a_kp1 * p1 + a_k * p2 + a_km1 * p3 + a_km2 * p4

            # Apply causal mask: zero out where k > q (column index > row index)
            causal_mask = k_offsets <= pid_q
            conv_val = tl.where(causal_mask & mask, conv_val, 0.0)

            # Store intermediate result
            tl.store(out_base + k_offsets * stride_ok, conv_val, mask=mask)

            # Accumulate sum for normalization (cast to float32 to match acc_sum type)
            block_sum = tl.sum(tl.where(mask, conv_val, 0.0).to(tl.float32))
            acc_sum = acc_sum + block_sum

        # Second pass: normalize
        # Avoid division by zero
        acc_sum = tl.maximum(acc_sum, 1e-9)

        for block_start in range(0, T_k, BLOCK_K):
            k_offsets = block_start + tl.arange(0, BLOCK_K)
            mask = k_offsets < T_k

            conv_val = tl.load(out_base + k_offsets * stride_ok, mask=mask, other=0.0)
            normalized = conv_val / acc_sum
            tl.store(out_base + k_offsets * stride_ok, normalized, mask=mask)


    class LookAroundConvFunction(torch.autograd.Function):
        """Autograd function for look-around convolution with Triton kernels."""

        @staticmethod
        def forward(ctx, attn_weights: torch.Tensor, proj_logits: torch.Tensor) -> torch.Tensor:
            B, H, T_q, T_k = attn_weights.shape

            # Compute softmax projection (small tensor, PyTorch is fine)
            proj_5 = F.softmax(proj_logits.float(), dim=-1)

            # Allocate output
            out = torch.empty_like(attn_weights)

            # Choose block size
            BLOCK_K = triton.next_power_of_2(min(T_k, 128))

            # Grid: one program per (batch * query, head)
            grid = (B * T_q, H)

            _look_around_conv_fwd_kernel[grid](
                attn_weights, proj_5, out,
                B, H, T_q, T_k,
                attn_weights.stride(0), attn_weights.stride(1),
                attn_weights.stride(2), attn_weights.stride(3),
                proj_5.stride(0), proj_5.stride(1),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                BLOCK_K=BLOCK_K,
            )

            # Save for backward
            ctx.save_for_backward(attn_weights, proj_logits, proj_5, out)
            ctx.B, ctx.H, ctx.T_q, ctx.T_k = B, H, T_q, T_k

            return out

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            attn_weights, proj_logits, proj_5, fwd_out = ctx.saved_tensors
            B, H, T_q, T_k = ctx.B, ctx.H, ctx.T_q, ctx.T_k

            grad_attn = None
            grad_proj = None

            # Use PyTorch autograd for both gradients for correctness
            # The Triton backward kernel approach was having issues with the complex
            # chain of operations (conv -> causal mask -> normalize)
            with torch.enable_grad():
                # Recompute forward with autograd
                attn_grad = attn_weights.detach().requires_grad_(True)
                proj_logits_grad = proj_logits.detach().requires_grad_(True)

                # Softmax projection
                proj_5_grad = F.softmax(proj_logits_grad, dim=-1)

                # Flip for convolution
                kernel = proj_5_grad.flip(-1).unsqueeze(1)

                # Apply convolution
                attn_flat = attn_grad.permute(0, 2, 1, 3).reshape(B * T_q, H, T_k)
                attn_conv = F.conv1d(attn_flat, kernel, padding=2, groups=H)
                attn_conv = attn_conv.reshape(B, T_q, H, T_k).permute(0, 2, 1, 3)

                # Apply causal mask
                row_idx = torch.arange(T_q, device=attn_weights.device).view(-1, 1)
                col_idx = torch.arange(T_k, device=attn_weights.device).view(1, -1)
                causal_mask = col_idx > row_idx
                attn_conv = attn_conv.masked_fill(causal_mask, 0.0)

                # Normalize
                attn_conv = attn_conv / attn_conv.sum(dim=-1, keepdim=True).clamp(min=1e-9)

                # Compute gradient
                loss = (attn_conv * grad_output.detach()).sum()
                loss.backward()

                if ctx.needs_input_grad[0]:
                    grad_attn = attn_grad.grad
                if ctx.needs_input_grad[1]:
                    grad_proj = proj_logits_grad.grad

            return grad_attn, grad_proj


    def look_around_conv_triton(
        attn_weights: torch.Tensor, proj_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Triton-accelerated look-around convolution.

        Args:
            attn_weights: (B, H, T_q, T_k) post-softmax attention weights
            proj_logits: (H, 5) learnable projection logits per head

        Returns:
            (B, H, T_q, T_k) convolved attention weights, summing to 1
        """
        return LookAroundConvFunction.apply(attn_weights, proj_logits)

else:
    # Triton not available, provide a stub that raises
    def look_around_conv_triton(
        attn_weights: torch.Tensor, proj_logits: torch.Tensor
    ) -> torch.Tensor:
        raise RuntimeError(
            "Triton is not available. Please install triton to use the Triton kernel, "
            "or use look_around_conv_pytorch instead."
        )
