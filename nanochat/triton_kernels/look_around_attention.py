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
        attn_ptr, proj_ptr, out_ptr, sums_ptr,
        B, H, T_q, T_k,
        stride_ab, stride_ah, stride_aq, stride_ak,
        stride_pb, stride_p5,
        stride_ob, stride_oh, stride_oq, stride_ok,
        stride_sb, stride_sh, stride_sq,
        BLOCK_K: tl.constexpr,
    ):
        """
        Forward kernel for look-around convolution (store-then-normalize strategy).

        Each program handles one (batch, head, query) combination.
        Pass 1: Compute convolution, store to output, accumulate sum
        Pass 2: Load from output, normalize, store back
        """
        pid_bq = tl.program_id(0)
        pid_h = tl.program_id(1)

        pid_b = pid_bq // T_q
        pid_q = pid_bq % T_q

        # Load projection weights into registers
        proj_base = proj_ptr + pid_h * stride_pb
        p0 = tl.load(proj_base + 0 * stride_p5)
        p1 = tl.load(proj_base + 1 * stride_p5)
        p2 = tl.load(proj_base + 2 * stride_p5)
        p3 = tl.load(proj_base + 3 * stride_p5)
        p4 = tl.load(proj_base + 4 * stride_p5)

        attn_base = attn_ptr + pid_b * stride_ab + pid_h * stride_ah + pid_q * stride_aq
        out_base = out_ptr + pid_b * stride_ob + pid_h * stride_oh + pid_q * stride_oq
        sums_base = sums_ptr + pid_b * stride_sb + pid_h * stride_sh + pid_q * stride_sq

        # Pass 1: compute convolution, store, accumulate sum
        acc_sum: tl.float32 = 0.0

        for block_start in range(0, T_k, BLOCK_K):
            k_offsets = block_start + tl.arange(0, BLOCK_K)
            mask = k_offsets < T_k

            # Load 5 neighbors with boundary handling
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

            # Apply causal mask
            is_causal = k_offsets <= pid_q
            conv_val = tl.where(is_causal & mask, conv_val, 0.0)

            # Store intermediate result
            tl.store(out_base + k_offsets * stride_ok, conv_val, mask=mask)

            # Accumulate sum
            acc_sum = acc_sum + tl.sum(tl.where(mask, conv_val, 0.0).to(tl.float32))

        # Store the sum for backward pass
        acc_sum = tl.maximum(acc_sum, 1e-9)
        tl.store(sums_base, acc_sum)

        # Pass 2: normalize (load from output, divide, store back)
        for block_start in range(0, T_k, BLOCK_K):
            k_offsets = block_start + tl.arange(0, BLOCK_K)
            mask = k_offsets < T_k

            conv_val = tl.load(out_base + k_offsets * stride_ok, mask=mask, other=0.0)
            normalized = conv_val / acc_sum
            tl.store(out_base + k_offsets * stride_ok, normalized, mask=mask)


    @triton.jit
    def _look_around_conv_fwd_kernel_recompute(
        attn_ptr, proj_ptr, out_ptr, sums_ptr,
        B, H, T_q, T_k,
        stride_ab, stride_ah, stride_aq, stride_ak,
        stride_pb, stride_p5,
        stride_ob, stride_oh, stride_oq, stride_ok,
        stride_sb, stride_sh, stride_sq,
        BLOCK_K: tl.constexpr,
    ):
        """
        Forward kernel for look-around convolution (recompute strategy).

        Trades compute for memory bandwidth:
        Pass 1: Compute convolution (no store), accumulate sum
        Pass 2: Recompute convolution, normalize, store (single write)

        L2 cache should serve the reloads effectively since we're reading
        the same attention values again immediately.
        """
        pid_bq = tl.program_id(0)
        pid_h = tl.program_id(1)

        pid_b = pid_bq // T_q
        pid_q = pid_bq % T_q

        # Load projection weights into registers
        proj_base = proj_ptr + pid_h * stride_pb
        p0 = tl.load(proj_base + 0 * stride_p5)
        p1 = tl.load(proj_base + 1 * stride_p5)
        p2 = tl.load(proj_base + 2 * stride_p5)
        p3 = tl.load(proj_base + 3 * stride_p5)
        p4 = tl.load(proj_base + 4 * stride_p5)

        attn_base = attn_ptr + pid_b * stride_ab + pid_h * stride_ah + pid_q * stride_aq
        out_base = out_ptr + pid_b * stride_ob + pid_h * stride_oh + pid_q * stride_oq
        sums_base = sums_ptr + pid_b * stride_sb + pid_h * stride_sh + pid_q * stride_sq

        # Pass 1: Accumulate sum only (no writes)
        acc_sum: tl.float32 = 0.0

        for block_start in range(0, T_k, BLOCK_K):
            k_offsets = block_start + tl.arange(0, BLOCK_K)
            mask = k_offsets < T_k

            # Load 5 neighbors
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
            val = a_kp2 * p0 + a_kp1 * p1 + a_k * p2 + a_km1 * p3 + a_km2 * p4

            # Apply causal mask
            is_causal = k_offsets <= pid_q
            val = tl.where(is_causal & mask, val, 0.0)

            # Accumulate sum (no store yet)
            acc_sum = acc_sum + tl.sum(val)

        # Store sum for backward pass
        acc_sum = tl.maximum(acc_sum, 1e-9)
        tl.store(sums_base, acc_sum)

        # Pass 2: Recompute & write (single write per element)
        for block_start in range(0, T_k, BLOCK_K):
            k_offsets = block_start + tl.arange(0, BLOCK_K)
            mask = k_offsets < T_k

            # Reload inputs (L2 cache should serve these)
            a_km2 = tl.load(attn_base + (k_offsets - 2) * stride_ak,
                           mask=mask & (k_offsets >= 2), other=0.0).to(tl.float32)
            a_km1 = tl.load(attn_base + (k_offsets - 1) * stride_ak,
                           mask=mask & (k_offsets >= 1), other=0.0).to(tl.float32)
            a_k = tl.load(attn_base + k_offsets * stride_ak, mask=mask, other=0.0).to(tl.float32)
            a_kp1 = tl.load(attn_base + (k_offsets + 1) * stride_ak,
                           mask=mask & (k_offsets + 1 < T_k), other=0.0).to(tl.float32)
            a_kp2 = tl.load(attn_base + (k_offsets + 2) * stride_ak,
                           mask=mask & (k_offsets + 2 < T_k), other=0.0).to(tl.float32)

            # Recompute convolution
            val = a_kp2 * p0 + a_kp1 * p1 + a_k * p2 + a_km1 * p3 + a_km2 * p4

            # Re-apply causal mask
            is_causal = k_offsets <= pid_q
            val = tl.where(is_causal & mask, val, 0.0)

            # Normalize and write (single write)
            val = val / acc_sum
            tl.store(out_base + k_offsets * stride_ok, val, mask=mask)


    @triton.jit
    def _look_around_conv_bwd_kernel(
        # Inputs
        attn_ptr, proj_ptr, out_ptr, grad_out_ptr, sums_ptr,
        # Outputs
        grad_attn_ptr, grad_proj_ptr,
        # Dimensions
        T_q, T_k,
        # Strides (attn/out/grad tensors share layout)
        stride_ab, stride_ah, stride_aq, stride_ak,
        # Strides for proj/grad_proj (both are H x 5)
        stride_ph, stride_p5,
        # Strides for sums
        stride_sb, stride_sh, stride_sq,
        # Constexpr
        BLOCK_K: tl.constexpr,
    ):
        """
        Backward kernel for look-around convolution.

        Optimized strategy:
        Pass 1: Compute D = dot(grad_out, out). (No convolution recompute needed!)
        Pass 2: Compute grad_attn via transposed convolution, accumulate grad_proj locally.

        Saves pre-computed normalization sums from forward pass to avoid recomputation.
        Uses atomic_add for grad_proj (efficient: only 5 atomics per thread block at the end).
        """
        pid_bq = tl.program_id(0)
        pid_h = tl.program_id(1)

        # Decode batch and query indices
        pid_b = pid_bq // T_q
        pid_q = pid_bq % T_q

        # Base pointers for this (batch, head, query) row
        offset_common = pid_b * stride_ab + pid_h * stride_ah + pid_q * stride_aq
        attn_base = attn_ptr + offset_common
        out_base = out_ptr + offset_common
        grad_out_base = grad_out_ptr + offset_common
        grad_attn_base = grad_attn_ptr + offset_common
        sums_base = sums_ptr + pid_b * stride_sb + pid_h * stride_sh + pid_q * stride_sq

        # Load projection weights into registers
        proj_base = proj_ptr + pid_h * stride_ph
        p0 = tl.load(proj_base + 0 * stride_p5).to(tl.float32)
        p1 = tl.load(proj_base + 1 * stride_p5).to(tl.float32)
        p2 = tl.load(proj_base + 2 * stride_p5).to(tl.float32)
        p3 = tl.load(proj_base + 3 * stride_p5).to(tl.float32)
        p4 = tl.load(proj_base + 4 * stride_p5).to(tl.float32)

        # Load pre-computed sum and avoid division by zero
        acc_sum = tl.load(sums_base)
        acc_sum = tl.maximum(acc_sum, 1e-9)

        # ===== Pass 1: Compute D = dot(grad_out, out) =====
        acc_D: tl.float32 = 0.0

        for block_start in range(0, T_k, BLOCK_K):
            k_offsets = block_start + tl.arange(0, BLOCK_K)
            mask = k_offsets < T_k

            out_val = tl.load(out_base + k_offsets * stride_ak, mask=mask, other=0.0).to(tl.float32)
            grad_out_val = tl.load(grad_out_base + k_offsets * stride_ak, mask=mask, other=0.0).to(tl.float32)
            acc_D += tl.sum(grad_out_val * out_val)

        # ===== Pass 2: Compute grad_attn and accumulate grad_proj =====
        dp0: tl.float32 = 0.0
        dp1: tl.float32 = 0.0
        dp2: tl.float32 = 0.0
        dp3: tl.float32 = 0.0
        dp4: tl.float32 = 0.0

        for block_start in range(0, T_k, BLOCK_K):
            k_offsets = block_start + tl.arange(0, BLOCK_K)
            mask = k_offsets < T_k
            is_causal = k_offsets <= pid_q

            # Compute grad_conv = (grad_out - D) / S
            grad_out_val = tl.load(grad_out_base + k_offsets * stride_ak, mask=mask, other=0.0).to(tl.float32)
            grad_conv = (grad_out_val - acc_D) / acc_sum
            grad_conv = tl.where(is_causal & mask, grad_conv, 0.0)

            # Load attention neighbors for grad_proj computation
            a_km2 = tl.load(attn_base + (k_offsets - 2) * stride_ak,
                           mask=mask & (k_offsets >= 2), other=0.0).to(tl.float32)
            a_km1 = tl.load(attn_base + (k_offsets - 1) * stride_ak,
                           mask=mask & (k_offsets >= 1), other=0.0).to(tl.float32)
            a_k = tl.load(attn_base + k_offsets * stride_ak, mask=mask, other=0.0).to(tl.float32)
            a_kp1 = tl.load(attn_base + (k_offsets + 1) * stride_ak,
                           mask=mask & (k_offsets + 1 < T_k), other=0.0).to(tl.float32)
            a_kp2 = tl.load(attn_base + (k_offsets + 2) * stride_ak,
                           mask=mask & (k_offsets + 2 < T_k), other=0.0).to(tl.float32)

            # Accumulate grad_proj: dp_i += sum(grad_conv * a_neighbor_i)
            dp0 += tl.sum(tl.where(mask, grad_conv * a_kp2, 0.0))
            dp1 += tl.sum(tl.where(mask, grad_conv * a_kp1, 0.0))
            dp2 += tl.sum(tl.where(mask, grad_conv * a_k, 0.0))
            dp3 += tl.sum(tl.where(mask, grad_conv * a_km1, 0.0))
            dp4 += tl.sum(tl.where(mask, grad_conv * a_km2, 0.0))

            # Load grad_out at neighbor positions for transposed convolution
            go_km2 = tl.load(grad_out_base + (k_offsets - 2) * stride_ak,
                            mask=mask & (k_offsets >= 2), other=0.0).to(tl.float32)
            go_km1 = tl.load(grad_out_base + (k_offsets - 1) * stride_ak,
                            mask=mask & (k_offsets >= 1), other=0.0).to(tl.float32)
            go_kp1 = tl.load(grad_out_base + (k_offsets + 1) * stride_ak,
                            mask=mask & (k_offsets + 1 < T_k), other=0.0).to(tl.float32)
            go_kp2 = tl.load(grad_out_base + (k_offsets + 2) * stride_ak,
                            mask=mask & (k_offsets + 2 < T_k), other=0.0).to(tl.float32)

            # Compute grad_conv at neighbor positions (with causal masking)
            is_causal_km2 = (k_offsets - 2) <= pid_q
            gc_km2 = tl.where(is_causal_km2 & (k_offsets >= 2), (go_km2 - acc_D) / acc_sum, 0.0)

            is_causal_km1 = (k_offsets - 1) <= pid_q
            gc_km1 = tl.where(is_causal_km1 & (k_offsets >= 1), (go_km1 - acc_D) / acc_sum, 0.0)

            gc_k = grad_conv

            is_causal_kp1 = (k_offsets + 1) <= pid_q
            gc_kp1 = tl.where(is_causal_kp1 & (k_offsets + 1 < T_k), (go_kp1 - acc_D) / acc_sum, 0.0)

            is_causal_kp2 = (k_offsets + 2) <= pid_q
            gc_kp2 = tl.where(is_causal_kp2 & (k_offsets + 2 < T_k), (go_kp2 - acc_D) / acc_sum, 0.0)

            # Transposed convolution for grad_attn
            grad_attn_val = gc_kp2 * p4 + gc_kp1 * p3 + gc_k * p2 + gc_km1 * p1 + gc_km2 * p0
            tl.store(grad_attn_base + k_offsets * stride_ak, grad_attn_val, mask=mask)

        # Atomic add to global grad_proj (efficient: only 5 atomics per thread block)
        grad_proj_base = grad_proj_ptr + pid_h * stride_ph
        tl.atomic_add(grad_proj_base + 0 * stride_p5, dp0)
        tl.atomic_add(grad_proj_base + 1 * stride_p5, dp1)
        tl.atomic_add(grad_proj_base + 2 * stride_p5, dp2)
        tl.atomic_add(grad_proj_base + 3 * stride_p5, dp3)
        tl.atomic_add(grad_proj_base + 4 * stride_p5, dp4)


    # Global flag to select kernel strategy
    _USE_RECOMPUTE_KERNEL = False


    class LookAroundConvFunction(torch.autograd.Function):
        """Autograd function for look-around convolution with Triton kernels."""

        @staticmethod
        def forward(ctx, attn_weights: torch.Tensor, proj_logits: torch.Tensor) -> torch.Tensor:
            B, H, T_q, T_k = attn_weights.shape

            # Compute softmax projection (small tensor, PyTorch is fine)
            proj_5 = F.softmax(proj_logits.float(), dim=-1)

            # Allocate output
            out = torch.empty_like(attn_weights)
            
            # Allocate sums buffer
            sums = torch.empty((B, H, T_q), device=attn_weights.device, dtype=torch.float32)

            # Choose block size
            BLOCK_K = triton.next_power_of_2(min(T_k, 128))

            # Grid: one program per (batch * query, head)
            grid = (B * T_q, H)

            # Select kernel based on global flag
            kernel = _look_around_conv_fwd_kernel_recompute if _USE_RECOMPUTE_KERNEL else _look_around_conv_fwd_kernel

            kernel[grid](
                attn_weights, proj_5, out, sums,
                B, H, T_q, T_k,
                attn_weights.stride(0), attn_weights.stride(1),
                attn_weights.stride(2), attn_weights.stride(3),
                proj_5.stride(0), proj_5.stride(1),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                sums.stride(0), sums.stride(1), sums.stride(2),
                BLOCK_K=BLOCK_K,
            )

            # Save for backward
            ctx.save_for_backward(attn_weights, proj_logits, proj_5, out, sums)
            ctx.B, ctx.H, ctx.T_q, ctx.T_k = B, H, T_q, T_k

            return out

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor):
            attn_weights, proj_logits, proj_5, fwd_out, sums = ctx.saved_tensors
            B, H, T_q, T_k = ctx.B, ctx.H, ctx.T_q, ctx.T_k

            grad_output = grad_output.contiguous()
            grad_attn = torch.empty_like(attn_weights)

            # Direct accumulation buffer for grad_proj (H, 5), initialized to zero
            # Using float32 for accumulation even with lower precision inputs
            grad_proj_5 = torch.zeros((H, 5), device=attn_weights.device, dtype=torch.float32)

            BLOCK_K = triton.next_power_of_2(min(T_k, 128))
            grid = (B * T_q, H)

            _look_around_conv_bwd_kernel[grid](
                attn_weights, proj_5, fwd_out, grad_output, sums,
                grad_attn, grad_proj_5,
                T_q, T_k,
                attn_weights.stride(0), attn_weights.stride(1),
                attn_weights.stride(2), attn_weights.stride(3),
                proj_5.stride(0), proj_5.stride(1),
                sums.stride(0), sums.stride(1), sums.stride(2),
                BLOCK_K=BLOCK_K,
            )

            # Softmax backward in PyTorch (small H x 5 tensor)
            if ctx.needs_input_grad[1]:
                # grad_proj_logits = proj_5 * (grad_proj_5 - dot(grad_proj_5, proj_5))
                dot_product = (grad_proj_5 * proj_5).sum(dim=-1, keepdim=True)
                grad_proj = proj_5 * (grad_proj_5 - dot_product)
                # Convert back to input dtype
                grad_proj = grad_proj.to(proj_logits.dtype)
            else:
                grad_proj = None

            if not ctx.needs_input_grad[0]:
                grad_attn = None

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


    def set_triton_kernel_strategy(use_recompute: bool = True):
        """
        Set the Triton kernel strategy for look-around convolution.

        Args:
            use_recompute: If True (default), use the recompute strategy that trades
                          compute for memory bandwidth (2x compute, 1x write).
                          If False, use the store-then-normalize strategy (1x compute, 2x write).
        """
        global _USE_RECOMPUTE_KERNEL
        _USE_RECOMPUTE_KERNEL = use_recompute


    def get_triton_kernel_strategy() -> str:
        """Get the current Triton kernel strategy name."""
        return "recompute" if _USE_RECOMPUTE_KERNEL else "store_normalize"


    def _run_kernel_directly(attn_weights, proj_5, use_recompute):
        """Run kernel directly without autograd overhead (for benchmarking)."""
        B, H, T_q, T_k = attn_weights.shape
        out = torch.empty_like(attn_weights)
        sums = torch.empty((B, H, T_q), device=attn_weights.device, dtype=torch.float32)
        BLOCK_K = triton.next_power_of_2(min(T_k, 128))
        grid = (B * T_q, H)

        kernel = _look_around_conv_fwd_kernel_recompute if use_recompute else _look_around_conv_fwd_kernel

        kernel[grid](
            attn_weights, proj_5, out, sums,
            B, H, T_q, T_k,
            attn_weights.stride(0), attn_weights.stride(1),
            attn_weights.stride(2), attn_weights.stride(3),
            proj_5.stride(0), proj_5.stride(1),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            sums.stride(0), sums.stride(1), sums.stride(2),
            BLOCK_K=BLOCK_K,
        )
        return out

else:
    # Triton not available, provide a stub that raises
    def look_around_conv_triton(
        attn_weights: torch.Tensor, proj_logits: torch.Tensor
    ) -> torch.Tensor:
        raise RuntimeError(
            "Triton is not available. Please install triton to use the Triton kernel, "
            "or use look_around_conv_pytorch instead."
        )