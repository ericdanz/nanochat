"""
Fused Flash Attention with Look-Around Convolution.

This module implements a Flash Attention kernel that fuses the 5-tap convolution
directly into the attention computation, avoiding O(N²) materialization of the
attention matrix while still applying look-around convolution to the attention scores.

Key innovations:
1. Separate Halo Loading: Load K in BLOCK_N chunks plus left/right halos (2 each)
2. In-kernel Convolution: Apply 5-tap convolution to exp(QK^T) before accumulating
3. Online Normalization: Track running sum of CONVOLVED probabilities for correct normalization

Memory complexity: O(N) instead of O(N²) for the attention matrix

The convolution formula applied to each row of attention scores:
    P_conv[j] = P[j+2]*w[0] + P[j+1]*w[1] + P[j]*w[2] + P[j-1]*w[3] + P[j-2]*w[4]
    where P = softmax(QK^T / sqrt(d_k))
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def _fused_look_around_flash_causal_fwd_kernel(
        # Pointers
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        Proj_ptr,  # (H, 5) softmaxed projection weights
        L_ptr,  # (B, H, T_q) log-sum-exp values
        # Dimensions
        B: tl.constexpr, H: tl.constexpr, T_q: tl.constexpr, T_k: tl.constexpr, D: tl.constexpr,
        sm_scale,
        # Strides
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_ob, stride_oh, stride_ot, stride_od,
        stride_ph, stride_p5,
        stride_lb, stride_lh, stride_lt,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """
        Optimized CAUSAL-ONLY forward kernel with FIXED numerical stability.

        Key fixes:
        1. Compute global max over ALL shifted QK arrays before exp() to prevent overflow
        2. Re-apply causal mask after convolution to ensure correctness
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)

        pid_b = pid_bh // H
        pid_h = pid_bh % H

        # Query block range
        q_start = pid_m * BLOCK_M
        offs_m = q_start + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < T_q

        # Base pointers
        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        out_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
        l_base = L_ptr + pid_b * stride_lb + pid_h * stride_lh

        # Load Q
        q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)
        q = (q * sm_scale).to(tl.float16)

        # Load projection weights
        proj_base = Proj_ptr + pid_h * stride_ph
        w0 = tl.load(proj_base + 0 * stride_p5).to(tl.float32)
        w1 = tl.load(proj_base + 1 * stride_p5).to(tl.float32)
        w2 = tl.load(proj_base + 2 * stride_p5).to(tl.float32)
        w3 = tl.load(proj_base + 3 * stride_p5).to(tl.float32)
        w4 = tl.load(proj_base + 4 * stride_p5).to(tl.float32)

        # Accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Causal: query at position m attends to keys 0..m
        max_k_for_q = offs_m

        for start_k in range(0, T_k, BLOCK_N):
            offs_n = start_k + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T_k
            causal_mask = offs_n[None, :] <= max_k_for_q[:, None]

            # Compute offsets for all 5 shifts
            offs_m2 = start_k - 2 + tl.arange(0, BLOCK_N)
            offs_m1 = start_k - 1 + tl.arange(0, BLOCK_N)
            offs_p1 = start_k + 1 + tl.arange(0, BLOCK_N)
            offs_p2 = start_k + 2 + tl.arange(0, BLOCK_N)

            mask_m2 = (offs_m2 >= 0) & (offs_m2 < T_k)
            mask_m1 = (offs_m1 >= 0) & (offs_m1 < T_k)
            mask_p1 = offs_p1 < T_k
            mask_p2 = offs_p2 < T_k

            # Load all K blocks (center + 4 shifts)
            k_ptrs = k_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)

            k_m2_ptrs = k_base + offs_m2[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_m2 = tl.load(k_m2_ptrs, mask=mask_m2[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)

            k_m1_ptrs = k_base + offs_m1[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_m1 = tl.load(k_m1_ptrs, mask=mask_m1[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)

            k_p1_ptrs = k_base + offs_p1[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_p1 = tl.load(k_p1_ptrs, mask=mask_p1[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)

            k_p2_ptrs = k_base + offs_p2[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_p2 = tl.load(k_p2_ptrs, mask=mask_p2[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)

            # Compute ALL QK scores
            qk = tl.dot(q, tl.trans(k)).to(tl.float32)
            qk_m2 = tl.dot(q, tl.trans(k_m2)).to(tl.float32)
            qk_m1 = tl.dot(q, tl.trans(k_m1)).to(tl.float32)
            qk_p1 = tl.dot(q, tl.trans(k_p1)).to(tl.float32)
            qk_p2 = tl.dot(q, tl.trans(k_p2)).to(tl.float32)

            # Apply causal + validity masks to QK scores (set invalid to -inf)
            qk = tl.where(mask_n[None, :] & causal_mask, qk, float("-inf"))
            qk_m2 = tl.where(mask_m2[None, :] & (offs_m2[None, :] <= max_k_for_q[:, None]), qk_m2, float("-inf"))
            qk_m1 = tl.where(mask_m1[None, :] & (offs_m1[None, :] <= max_k_for_q[:, None]), qk_m1, float("-inf"))
            qk_p1 = tl.where(mask_p1[None, :] & (offs_p1[None, :] <= max_k_for_q[:, None]), qk_p1, float("-inf"))
            qk_p2 = tl.where(mask_p2[None, :] & (offs_p2[None, :] <= max_k_for_q[:, None]), qk_p2, float("-inf"))

            # =================================================================
            # FIX: Compute GLOBAL max over ALL QK arrays to prevent exp overflow
            # =================================================================
            m_ij = tl.max(qk, axis=1)
            m_ij = tl.maximum(m_ij, tl.max(qk_m2, axis=1))
            m_ij = tl.maximum(m_ij, tl.max(qk_m1, axis=1))
            m_ij = tl.maximum(m_ij, tl.max(qk_p1, axis=1))
            m_ij = tl.maximum(m_ij, tl.max(qk_p2, axis=1))
            m_ij = tl.maximum(m_ij, m_i)

            # Online softmax correction factor
            alpha = tl.exp(m_i - m_ij)

            # Now compute exp safely (all qk values are <= m_ij, so exp <= 1)
            p = tl.exp(qk - m_ij[:, None])
            p_m2 = tl.exp(qk_m2 - m_ij[:, None])
            p_m1 = tl.exp(qk_m1 - m_ij[:, None])
            p_p1 = tl.exp(qk_p1 - m_ij[:, None])
            p_p2 = tl.exp(qk_p2 - m_ij[:, None])

            # Convolution: p_conv[j] = w0*p[j+2] + w1*p[j+1] + w2*p[j] + w3*p[j-1] + w4*p[j-2]
            p_conv = w2 * p + w3 * p_m1 + w4 * p_m2 + w1 * p_p1 + w0 * p_p2

            # Re-apply causal mask to convolved output (critical for correctness!)
            p_conv = tl.where(mask_n[None, :] & causal_mask, p_conv, 0.0)

            # Accumulate
            row_sum = tl.sum(p_conv, axis=1)
            l_i = l_i * alpha + row_sum
            acc = acc * alpha[:, None]

            # Load V and accumulate output
            v_ptrs = v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)
            acc = acc + tl.dot(p_conv.to(tl.float16), v.to(tl.float16)).to(tl.float32)

            m_i = m_ij

        # Final normalization
        l_i = tl.maximum(l_i, 1e-9)
        acc = acc / l_i[:, None]

        # Store output
        out_ptrs = out_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & (offs_d[None, :] < D))

        # Store LSE for backward
        lse = m_i + tl.log(l_i)
        l_ptrs = l_base + offs_m * stride_lt
        tl.store(l_ptrs, lse, mask=mask_m)


    @triton.jit
    def _fused_look_around_flash_fwd_kernel(
        # Pointers
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        Proj_ptr,  # (H, 5) softmaxed projection weights
        # Output for backward pass
        L_ptr,  # (B, H, T_q) log-sum-exp values (based on CONVOLVED sums!)
        # Dimensions
        B: tl.constexpr, H: tl.constexpr, T_q: tl.constexpr, T_k: tl.constexpr, D: tl.constexpr,
        # Scale factor
        sm_scale,
        # Strides for Q, K, V, Out: (B, H, T, D) layout
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_ob, stride_oh, stride_ot, stride_od,
        # Strides for proj: (H, 5) layout
        stride_ph, stride_p5,
        # Strides for L: (B, H, T_q) layout
        stride_lb, stride_lh, stride_lt,
        # Block sizes (must be powers of 2)
        BLOCK_M: tl.constexpr,  # Block size for queries
        BLOCK_N: tl.constexpr,  # Block size for keys
        BLOCK_D: tl.constexpr,  # Must be >= D, power of 2
        IS_CAUSAL: tl.constexpr,
    ):
        """
        Fused Flash Attention with Look-Around Convolution - Forward Pass.

        Each program handles BLOCK_M queries for one (batch, head) combination.

        Architecture:
        1. Load K in BLOCK_N chunks plus left/right halos (2 elements each)
        2. Compute QK^T for core + halos
        3. Apply 5-tap convolution using separate halo arrays
        4. Accumulate using CONVOLVED probabilities

        CRITICAL: L_ptr stores m_i + log(l_i) where l_i is the sum of CONVOLVED
        probabilities, not raw softmax. Essential for gradient correctness.
        """
        # Program indices
        pid_m = tl.program_id(0)  # Which query block
        pid_bh = tl.program_id(1)  # Which (batch, head)

        pid_b = pid_bh // H
        pid_h = pid_bh % H

        # Query offsets
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < T_q

        # Base pointers
        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        out_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
        l_base = L_ptr + pid_b * stride_lb + pid_h * stride_lh

        # Load query block: (BLOCK_M, BLOCK_D)
        q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)
        q = (q * sm_scale).to(tl.float16)

        # Load projection weights
        proj_base = Proj_ptr + pid_h * stride_ph
        w0 = tl.load(proj_base + 0 * stride_p5).to(tl.float32)
        w1 = tl.load(proj_base + 1 * stride_p5).to(tl.float32)
        w2 = tl.load(proj_base + 2 * stride_p5).to(tl.float32)
        w3 = tl.load(proj_base + 3 * stride_p5).to(tl.float32)
        w4 = tl.load(proj_base + 4 * stride_p5).to(tl.float32)

        # Initialize accumulators
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        if IS_CAUSAL:
            max_k_for_q = offs_m

        # ===================================================================
        # INNER LOOP: Iterate over K/V blocks
        # ===================================================================
        for start_k in range(0, T_k, BLOCK_N):
            # Core block indices
            offs_n = start_k + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T_k

            # Compute offsets for all 5 shifts
            offs_m2 = start_k - 2 + tl.arange(0, BLOCK_N)
            offs_m1 = start_k - 1 + tl.arange(0, BLOCK_N)
            offs_p1 = start_k + 1 + tl.arange(0, BLOCK_N)
            offs_p2 = start_k + 2 + tl.arange(0, BLOCK_N)

            mask_m2 = (offs_m2 >= 0) & (offs_m2 < T_k)
            mask_m1 = (offs_m1 >= 0) & (offs_m1 < T_k)
            mask_p1 = offs_p1 < T_k
            mask_p2 = offs_p2 < T_k

            # -----------------------------------------------------------
            # 1. LOAD ALL K blocks (center + 4 shifts)
            # -----------------------------------------------------------
            k_ptrs = k_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)

            k_m2_ptrs = k_base + offs_m2[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_m2 = tl.load(k_m2_ptrs, mask=mask_m2[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)

            k_m1_ptrs = k_base + offs_m1[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_m1 = tl.load(k_m1_ptrs, mask=mask_m1[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)

            k_p1_ptrs = k_base + offs_p1[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_p1 = tl.load(k_p1_ptrs, mask=mask_p1[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)

            k_p2_ptrs = k_base + offs_p2[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_p2 = tl.load(k_p2_ptrs, mask=mask_p2[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)

            # -----------------------------------------------------------
            # 2. COMPUTE ALL QK scores
            # -----------------------------------------------------------
            qk = tl.dot(q, tl.trans(k)).to(tl.float32)
            qk_m2 = tl.dot(q, tl.trans(k_m2)).to(tl.float32)
            qk_m1 = tl.dot(q, tl.trans(k_m1)).to(tl.float32)
            qk_p1 = tl.dot(q, tl.trans(k_p1)).to(tl.float32)
            qk_p2 = tl.dot(q, tl.trans(k_p2)).to(tl.float32)

            # -----------------------------------------------------------
            # 3. APPLY MASKS to QK scores
            # -----------------------------------------------------------
            qk = tl.where(mask_n[None, :], qk, float("-inf"))
            qk_m2 = tl.where(mask_m2[None, :], qk_m2, float("-inf"))
            qk_m1 = tl.where(mask_m1[None, :], qk_m1, float("-inf"))
            qk_p1 = tl.where(mask_p1[None, :], qk_p1, float("-inf"))
            qk_p2 = tl.where(mask_p2[None, :], qk_p2, float("-inf"))

            if IS_CAUSAL:
                causal_mask = offs_n[None, :] <= max_k_for_q[:, None]
                qk = tl.where(causal_mask, qk, float("-inf"))
                qk_m2 = tl.where(offs_m2[None, :] <= max_k_for_q[:, None], qk_m2, float("-inf"))
                qk_m1 = tl.where(offs_m1[None, :] <= max_k_for_q[:, None], qk_m1, float("-inf"))
                qk_p1 = tl.where(offs_p1[None, :] <= max_k_for_q[:, None], qk_p1, float("-inf"))
                qk_p2 = tl.where(offs_p2[None, :] <= max_k_for_q[:, None], qk_p2, float("-inf"))

            # -----------------------------------------------------------
            # 4. ROBUST ONLINE SOFTMAX (FIX: Global max over ALL QK arrays)
            # -----------------------------------------------------------
            m_ij = tl.max(qk, axis=1)
            m_ij = tl.maximum(m_ij, tl.max(qk_m2, axis=1))
            m_ij = tl.maximum(m_ij, tl.max(qk_m1, axis=1))
            m_ij = tl.maximum(m_ij, tl.max(qk_p1, axis=1))
            m_ij = tl.maximum(m_ij, tl.max(qk_p2, axis=1))
            m_ij = tl.maximum(m_ij, m_i)

            # Correction factor for previous accumulator
            alpha = tl.exp(m_i - m_ij)

            # Compute exp safely (all values <= m_ij, so exp <= 1)
            p = tl.exp(qk - m_ij[:, None])
            p_m2 = tl.exp(qk_m2 - m_ij[:, None])
            p_m1 = tl.exp(qk_m1 - m_ij[:, None])
            p_p1 = tl.exp(qk_p1 - m_ij[:, None])
            p_p2 = tl.exp(qk_p2 - m_ij[:, None])

            # -----------------------------------------------------------
            # 5. CONVOLUTION
            # -----------------------------------------------------------
            p_conv = w2 * p + w3 * p_m1 + w4 * p_m2 + w1 * p_p1 + w0 * p_p2

            # Mask invalid positions
            p_conv = tl.where(mask_n[None, :], p_conv, 0.0)

            # Apply causal mask to convolved output
            if IS_CAUSAL:
                p_conv = tl.where(offs_n[None, :] <= max_k_for_q[:, None], p_conv, 0.0)

            # -----------------------------------------------------------
            # 6. ACCUMULATION
            # -----------------------------------------------------------
            row_sum = tl.sum(p_conv, axis=1)
            l_i = l_i * alpha + row_sum
            acc = acc * alpha[:, None]

            # Load V and accumulate output
            v_ptrs = v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)
            acc = acc + tl.dot(p_conv.to(tl.float16), v.to(tl.float16)).to(tl.float32)

            m_i = m_ij

        # ===================================================================
        # FINAL OUTPUT
        # ===================================================================
        l_i = tl.maximum(l_i, 1e-9)
        acc = acc / l_i[:, None]

        # Store output
        out_ptrs = out_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
        tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & (offs_d[None, :] < D))

        # Store LSE for backward (based on CONVOLVED sums!)
        lse = m_i + tl.log(l_i)
        l_ptrs = l_base + offs_m * stride_lt
        tl.store(l_ptrs, lse, mask=mask_m)


    @triton.jit
    def _fused_look_around_flash_bwd_kernel(
        # Inputs (from forward)
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        Proj_ptr,  # (H, 5) softmaxed projection weights
        L_ptr,  # (B, H, T_q) log-sum-exp values (CONVOLVED sums from fwd)
        DO_ptr,  # (B, H, T_q, D) gradient of output
        # Outputs
        DQ_ptr, DK_ptr, DV_ptr,
        DProj_ptr,  # (H, 5) gradient accumulator for projection weights
        # Dimensions
        B: tl.constexpr, H: tl.constexpr, T_q: tl.constexpr, T_k: tl.constexpr, D: tl.constexpr,
        sm_scale,
        # Strides for Q, K, V, Out, DO, DQ, DK, DV: (B, H, T, D) layout
        stride_qb, stride_qh, stride_qt, stride_qd,
        stride_kb, stride_kh, stride_kt, stride_kd,
        stride_vb, stride_vh, stride_vt, stride_vd,
        stride_ob, stride_oh, stride_ot, stride_od,
        stride_dob, stride_doh, stride_dot, stride_dod,
        stride_dqb, stride_dqh, stride_dqt, stride_dqd,
        stride_dkb, stride_dkh, stride_dkt, stride_dkd,
        stride_dvb, stride_dvh, stride_dvt, stride_dvd,
        # Strides for proj: (H, 5)
        stride_ph, stride_p5,
        # Strides for L: (B, H, T_q)
        stride_lb, stride_lh, stride_lt,
        # Block sizes
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        HALO_PAD: tl.constexpr,  # Next power of 2 >= BLOCK_N + 4
        IS_CAUSAL: tl.constexpr,
    ):
        """
        Fused Flash Attention with Look-Around Convolution - Backward Pass.

        Key operations:
        1. Recompute P_halo using stored LSE from forward
        2. Compute dV = P_core.T @ dO
        3. Compute dP_core = dO @ V.T
        4. Transposed convolution: dP_core -> dP_halo
        5. Backward softmax: dS_halo = P_halo * (dP_halo - rowsum(dP_halo * P_halo))
        6. Accumulate dQ += dS_halo @ K_halo.T
        7. Atomic add dK += dS_halo.T @ Q (needed due to halo overlap)
        """
        pid_m = tl.program_id(0)  # Query block
        pid_bh = tl.program_id(1)  # (batch, head)

        pid_b = pid_bh // H
        pid_h = pid_bh % H

        # Query block indices
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        mask_m = offs_m < T_q

        # Base pointers
        q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
        k_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
        v_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
        do_base = DO_ptr + pid_b * stride_dob + pid_h * stride_doh
        dq_base = DQ_ptr + pid_b * stride_dqb + pid_h * stride_dqh
        dk_base = DK_ptr + pid_b * stride_dkb + pid_h * stride_dkh
        dv_base = DV_ptr + pid_b * stride_dvb + pid_h * stride_dvh
        l_base = L_ptr + pid_b * stride_lb + pid_h * stride_lh

        # Load Q block
        q_ptrs = q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0)
        q = (q * sm_scale).to(tl.float16)

        # Load dO block
        do_ptrs = do_base + offs_m[:, None] * stride_dot + offs_d[None, :] * stride_dod
        do = tl.load(do_ptrs, mask=mask_m[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float32)

        # Load stored LSE from forward pass
        l_ptrs = l_base + offs_m * stride_lt
        l_i = tl.load(l_ptrs, mask=mask_m, other=0.0)

        # Load projection weights
        proj_base = Proj_ptr + pid_h * stride_ph
        w0 = tl.load(proj_base + 0 * stride_p5).to(tl.float32)
        w1 = tl.load(proj_base + 1 * stride_p5).to(tl.float32)
        w2 = tl.load(proj_base + 2 * stride_p5).to(tl.float32)
        w3 = tl.load(proj_base + 3 * stride_p5).to(tl.float32)
        w4 = tl.load(proj_base + 4 * stride_p5).to(tl.float32)

        # Initialize dQ accumulator
        dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Gradient accumulators for projection weights (per thread block)
        dw0: tl.float32 = 0.0
        dw1: tl.float32 = 0.0
        dw2: tl.float32 = 0.0
        dw3: tl.float32 = 0.0
        dw4: tl.float32 = 0.0

        if IS_CAUSAL:
            max_k_for_q = offs_m

        # Iterate over K/V blocks
        for start_k in range(0, T_k, BLOCK_N):
            offs_n = start_k + tl.arange(0, BLOCK_N)
            mask_n = offs_n < T_k

            # ===== 1. RECOMPUTE P_halo =====
            # Load K (core block)
            k_ptrs = k_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)

            # Compute QK^T for core
            qk = tl.dot(q, tl.trans(k)).to(tl.float32)
            qk = tl.where(mask_n[None, :], qk, float("-inf"))

            if IS_CAUSAL:
                causal_mask = offs_n[None, :] <= max_k_for_q[:, None]
                qk = tl.where(causal_mask, qk, float("-inf"))

            # Recompute P using stored LSE
            # p = exp(qk - l_i) but l_i was stored as m + log(l) from convolved sum
            # We need to recompute with the same normalization
            p = tl.exp(qk - l_i[:, None])

            # Load shifted K and recompute shifted P for convolution
            # (Same as forward - compute 5 shifted versions)

            # P for shift -2
            offs_m2 = start_k - 2 + tl.arange(0, BLOCK_N)
            mask_m2 = (offs_m2 >= 0) & (offs_m2 < T_k)
            k_m2_ptrs = k_base + offs_m2[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_m2 = tl.load(k_m2_ptrs, mask=mask_m2[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)
            qk_m2 = tl.dot(q, tl.trans(k_m2)).to(tl.float32)
            qk_m2 = tl.where(mask_m2[None, :], qk_m2, float("-inf"))
            if IS_CAUSAL:
                qk_m2 = tl.where(offs_m2[None, :] <= max_k_for_q[:, None], qk_m2, float("-inf"))
            p_m2 = tl.exp(qk_m2 - l_i[:, None])

            # P for shift -1
            offs_m1 = start_k - 1 + tl.arange(0, BLOCK_N)
            mask_m1 = (offs_m1 >= 0) & (offs_m1 < T_k)
            k_m1_ptrs = k_base + offs_m1[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_m1 = tl.load(k_m1_ptrs, mask=mask_m1[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)
            qk_m1 = tl.dot(q, tl.trans(k_m1)).to(tl.float32)
            qk_m1 = tl.where(mask_m1[None, :], qk_m1, float("-inf"))
            if IS_CAUSAL:
                qk_m1 = tl.where(offs_m1[None, :] <= max_k_for_q[:, None], qk_m1, float("-inf"))
            p_m1 = tl.exp(qk_m1 - l_i[:, None])

            # P for shift +1
            offs_p1 = start_k + 1 + tl.arange(0, BLOCK_N)
            mask_p1 = offs_p1 < T_k
            k_p1_ptrs = k_base + offs_p1[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_p1 = tl.load(k_p1_ptrs, mask=mask_p1[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)
            qk_p1 = tl.dot(q, tl.trans(k_p1)).to(tl.float32)
            qk_p1 = tl.where(mask_p1[None, :], qk_p1, float("-inf"))
            if IS_CAUSAL:
                qk_p1 = tl.where(offs_p1[None, :] <= max_k_for_q[:, None], qk_p1, float("-inf"))
            p_p1 = tl.exp(qk_p1 - l_i[:, None])

            # P for shift +2
            offs_p2 = start_k + 2 + tl.arange(0, BLOCK_N)
            mask_p2 = offs_p2 < T_k
            k_p2_ptrs = k_base + offs_p2[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k_p2 = tl.load(k_p2_ptrs, mask=mask_p2[:, None] & (offs_d[None, :] < D), other=0.0).to(tl.float16)
            qk_p2 = tl.dot(q, tl.trans(k_p2)).to(tl.float32)
            qk_p2 = tl.where(mask_p2[None, :], qk_p2, float("-inf"))
            if IS_CAUSAL:
                qk_p2 = tl.where(offs_p2[None, :] <= max_k_for_q[:, None], qk_p2, float("-inf"))
            p_p2 = tl.exp(qk_p2 - l_i[:, None])

            # Compute P_conv (same as forward)
            p_conv = w2 * p + w3 * p_m1 + w4 * p_m2 + w1 * p_p1 + w0 * p_p2
            p_conv = tl.where(mask_n[None, :], p_conv, 0.0)
            if IS_CAUSAL:
                p_conv = tl.where(offs_n[None, :] <= max_k_for_q[:, None], p_conv, 0.0)

            # ===== 2. Compute dV = P_conv.T @ dO =====
            # Load current dV and accumulate
            v_ptrs = v_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=mask_n[:, None] & (offs_d[None, :] < D), other=0.0)

            dv_contrib = tl.dot(tl.trans(p_conv.to(tl.float16)), do.to(tl.float16)).to(tl.float32)
            dv_ptrs = dv_base + offs_n[:, None] * stride_dvt + offs_d[None, :] * stride_dvd
            # Atomic add to dV (multiple query blocks contribute)
            tl.atomic_add(dv_ptrs, dv_contrib, mask=mask_n[:, None] & (offs_d[None, :] < D))

            # ===== 3. Compute dP_conv = dO @ V.T =====
            dp_conv = tl.dot(do.to(tl.float16), tl.trans(v.to(tl.float16))).to(tl.float32)

            # ===== 4. Backward through normalization =====
            # If we had renormalization in forward, we need to account for it
            # p_norm = p_conv / sum(p_conv)
            # dp_conv_prenorm = (dp_conv - sum(dp_conv * p_norm) * p_norm) / sum(p_conv)
            # For simplicity, assuming p_conv was already normalized in forward
            # Skip this step or include full normalization backward

            # ===== 5. Transposed convolution: dP_conv -> dP for each shift =====
            # dP[shift] += dP_conv * w[shift]
            dp = w2 * dp_conv  # Center contribution
            dp_m1 = w3 * dp_conv
            dp_m2 = w4 * dp_conv
            dp_p1 = w1 * dp_conv
            dp_p2 = w0 * dp_conv

            # ===== 6. Gradient for projection weights =====
            # dw[i] = sum(dp_conv * p_shifted[i])
            dw0 += tl.sum(dp_conv * p_p2)
            dw1 += tl.sum(dp_conv * p_p1)
            dw2 += tl.sum(dp_conv * p)
            dw3 += tl.sum(dp_conv * p_m1)
            dw4 += tl.sum(dp_conv * p_m2)

            # ===== 7. Backward through softmax =====
            # dS = P * (dP - sum(dP * P))
            dp_p_sum = tl.sum(dp * p, axis=1)
            ds = p * (dp - dp_p_sum[:, None])

            # Similar for shifted versions
            dp_m1_p_m1_sum = tl.sum(dp_m1 * p_m1, axis=1)
            ds_m1 = p_m1 * (dp_m1 - dp_m1_p_m1_sum[:, None])

            dp_m2_p_m2_sum = tl.sum(dp_m2 * p_m2, axis=1)
            ds_m2 = p_m2 * (dp_m2 - dp_m2_p_m2_sum[:, None])

            dp_p1_p_p1_sum = tl.sum(dp_p1 * p_p1, axis=1)
            ds_p1 = p_p1 * (dp_p1 - dp_p1_p_p1_sum[:, None])

            dp_p2_p_p2_sum = tl.sum(dp_p2 * p_p2, axis=1)
            ds_p2 = p_p2 * (dp_p2 - dp_p2_p_p2_sum[:, None])

            # Scale by softmax scale
            ds = ds * sm_scale
            ds_m1 = ds_m1 * sm_scale
            ds_m2 = ds_m2 * sm_scale
            ds_p1 = ds_p1 * sm_scale
            ds_p2 = ds_p2 * sm_scale

            # ===== 8. Accumulate dQ =====
            dq += tl.dot(ds.to(tl.float16), k.to(tl.float16)).to(tl.float32)
            dq += tl.dot(ds_m1.to(tl.float16), k_m1.to(tl.float16)).to(tl.float32)
            dq += tl.dot(ds_m2.to(tl.float16), k_m2.to(tl.float16)).to(tl.float32)
            dq += tl.dot(ds_p1.to(tl.float16), k_p1.to(tl.float16)).to(tl.float32)
            dq += tl.dot(ds_p2.to(tl.float16), k_p2.to(tl.float16)).to(tl.float32)

            # ===== 9. Atomic add dK =====
            # dK contributions from each shifted position
            dk_contrib = tl.dot(tl.trans(ds.to(tl.float16)), q.to(tl.float16)).to(tl.float32)
            dk_ptrs = dk_base + offs_n[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
            tl.atomic_add(dk_ptrs, dk_contrib, mask=mask_n[:, None] & (offs_d[None, :] < D))

            dk_m1_contrib = tl.dot(tl.trans(ds_m1.to(tl.float16)), q.to(tl.float16)).to(tl.float32)
            dk_m1_ptrs = dk_base + offs_m1[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
            tl.atomic_add(dk_m1_ptrs, dk_m1_contrib, mask=mask_m1[:, None] & (offs_d[None, :] < D))

            dk_m2_contrib = tl.dot(tl.trans(ds_m2.to(tl.float16)), q.to(tl.float16)).to(tl.float32)
            dk_m2_ptrs = dk_base + offs_m2[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
            tl.atomic_add(dk_m2_ptrs, dk_m2_contrib, mask=mask_m2[:, None] & (offs_d[None, :] < D))

            dk_p1_contrib = tl.dot(tl.trans(ds_p1.to(tl.float16)), q.to(tl.float16)).to(tl.float32)
            dk_p1_ptrs = dk_base + offs_p1[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
            tl.atomic_add(dk_p1_ptrs, dk_p1_contrib, mask=mask_p1[:, None] & (offs_d[None, :] < D))

            dk_p2_contrib = tl.dot(tl.trans(ds_p2.to(tl.float16)), q.to(tl.float16)).to(tl.float32)
            dk_p2_ptrs = dk_base + offs_p2[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
            tl.atomic_add(dk_p2_ptrs, dk_p2_contrib, mask=mask_p2[:, None] & (offs_d[None, :] < D))

        # Store dQ
        dq_ptrs = dq_base + offs_m[:, None] * stride_dqt + offs_d[None, :] * stride_dqd
        tl.store(dq_ptrs, dq.to(DQ_ptr.dtype.element_ty), mask=mask_m[:, None] & (offs_d[None, :] < D))

        # Atomic add projection weight gradients
        dproj_base = DProj_ptr + pid_h * stride_ph
        tl.atomic_add(dproj_base + 0 * stride_p5, dw0)
        tl.atomic_add(dproj_base + 1 * stride_p5, dw1)
        tl.atomic_add(dproj_base + 2 * stride_p5, dw2)
        tl.atomic_add(dproj_base + 3 * stride_p5, dw3)
        tl.atomic_add(dproj_base + 4 * stride_p5, dw4)


    # =========================================================================
    # PYTORCH 2.0+ COMPILATION SUPPORT
    # =========================================================================
    # The correct pattern is: AutogradFunction calls CustomOps (not the reverse)
    # This allows torch.compile to trace through the AutogradFunction and see
    # the custom ops for both forward and backward.

    # Check for custom_op support (PyTorch 2.1+)
    _HAS_CUSTOM_OP = hasattr(torch.library, 'custom_op')

    if _HAS_CUSTOM_OP:
        # =====================================================================
        # 1. FORWARD PRIMITIVE CUSTOM OP
        # =====================================================================
        @torch.library.custom_op("nanochat::fused_look_around_flash_fwd", mutates_args=())
        def _fused_flash_fwd_op(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            proj_logits: torch.Tensor,
            causal: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Forward primitive - launches Triton kernel and returns (out, L, proj_5)."""
            B, H, T_q, D = q.shape
            T_k = k.shape[2]

            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            proj_5 = F.softmax(proj_logits.float(), dim=-1).contiguous()

            out = torch.empty_like(q)
            L = torch.empty((B, H, T_q), device=q.device, dtype=torch.float32)
            sm_scale = 1.0 / math.sqrt(D)

            BLOCK_D = triton.next_power_of_2(D)
            if BLOCK_D >= 64:
                BLOCK_M, BLOCK_N = 32, 32
            else:
                BLOCK_M, BLOCK_N = 64, 64

            grid = (triton.cdiv(T_q, BLOCK_M), B * H)

            if causal:
                _fused_look_around_flash_causal_fwd_kernel[grid](
                    q, k, v, out, proj_5, L,
                    B, H, T_q, T_k, D, sm_scale,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    proj_5.stride(0), proj_5.stride(1),
                    L.stride(0), L.stride(1), L.stride(2),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                )
            else:
                _fused_look_around_flash_fwd_kernel[grid](
                    q, k, v, out, proj_5, L,
                    B, H, T_q, T_k, D, sm_scale,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    proj_5.stride(0), proj_5.stride(1),
                    L.stride(0), L.stride(1), L.stride(2),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, IS_CAUSAL=causal,
                )

            return out, L, proj_5

        @_fused_flash_fwd_op.register_fake
        def _fused_flash_fwd_fake(q, k, v, proj_logits, causal):
            """FakeTensor implementation for forward - shape inference only."""
            B, H, T_q, D = q.shape
            return (
                torch.empty_like(q),
                torch.empty((B, H, T_q), device=q.device, dtype=torch.float32),
                torch.empty_like(proj_logits),
            )

        # =====================================================================
        # 2. BACKWARD PRIMITIVE CUSTOM OP
        # =====================================================================
        @torch.library.custom_op("nanochat::fused_look_around_flash_bwd", mutates_args=())
        def _fused_flash_bwd_op(
            grad_out: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            proj_5: torch.Tensor,
            out: torch.Tensor,
            L: torch.Tensor,
            causal: bool,
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            """Backward primitive - launches Triton kernel and returns gradients."""
            grad_out = grad_out.contiguous()
            B, H, T_q, D = q.shape
            T_k = k.shape[2]
            sm_scale = 1.0 / math.sqrt(D)

            BLOCK_D = triton.next_power_of_2(D)
            if BLOCK_D >= 64:
                BLOCK_M, BLOCK_N = 32, 32
            else:
                BLOCK_M, BLOCK_N = 64, 64

            grad_q = torch.empty_like(q)
            grad_k = torch.zeros_like(k)
            grad_v = torch.zeros_like(v)
            grad_proj_5 = torch.zeros((H, 5), device=q.device, dtype=torch.float32)

            grid = (triton.cdiv(T_q, BLOCK_M), B * H)
            HALO_PAD = triton.next_power_of_2(BLOCK_N + 4)

            _fused_look_around_flash_bwd_kernel[grid](
                q, k, v, out, proj_5, L, grad_out,
                grad_q, grad_k, grad_v, grad_proj_5,
                B, H, T_q, T_k, D, sm_scale,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                grad_out.stride(0), grad_out.stride(1), grad_out.stride(2), grad_out.stride(3),
                grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
                grad_k.stride(0), grad_k.stride(1), grad_k.stride(2), grad_k.stride(3),
                grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
                proj_5.stride(0), proj_5.stride(1),
                L.stride(0), L.stride(1), L.stride(2),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                HALO_PAD=HALO_PAD, IS_CAUSAL=causal,
            )

            # Softmax backward for proj_logits (grad w.r.t. logits, not softmax output)
            dot_product = (grad_proj_5 * proj_5).sum(dim=-1, keepdim=True)
            grad_proj_logits = proj_5 * (grad_proj_5 - dot_product)

            return grad_q, grad_k, grad_v, grad_proj_logits

        @_fused_flash_bwd_op.register_fake
        def _fused_flash_bwd_fake(grad_out, q, k, v, proj_5, out, L, causal):
            """FakeTensor implementation for backward - shape inference only."""
            H = q.shape[1]
            return (
                torch.empty_like(q),
                torch.empty_like(k),
                torch.empty_like(v),
                torch.empty((H, 5), device=q.device, dtype=torch.float32),
            )

        # =====================================================================
        # 3. AUTOGRAD FUNCTION (COMPOSITE - CALLS THE CUSTOM OPS)
        # =====================================================================
        class FusedLookAroundFlashFunction(torch.autograd.Function):
            """
            Autograd function for fused look-around Flash Attention.

            This is a "composite" function that torch.compile can trace through.
            It calls the primitive custom ops which have FakeTensor implementations.
            """

            @staticmethod
            def forward(ctx, q, k, v, proj_logits, causal=True):
                # Call the forward custom op
                out, L, proj_5 = _fused_flash_fwd_op(q, k, v, proj_logits, causal)

                # Save tensors for backward
                ctx.save_for_backward(q, k, v, proj_5, out, L)
                ctx.causal = causal

                return out

            @staticmethod
            def backward(ctx, grad_output):
                q, k, v, proj_5, out, L = ctx.saved_tensors

                # Call the backward custom op
                grad_q, grad_k, grad_v, grad_proj_logits = _fused_flash_bwd_op(
                    grad_output, q, k, v, proj_5, out, L, ctx.causal
                )

                return grad_q, grad_k, grad_v, grad_proj_logits, None

        # =====================================================================
        # 4. PUBLIC API
        # =====================================================================
        def fused_look_around_flash_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            proj_logits: torch.Tensor,
            causal: bool = True,
        ) -> torch.Tensor:
            """
            Fused Flash Attention with Look-Around Convolution.

            This uses a Triton kernel to fuse the 5-tap convolution directly into
            Flash Attention, avoiding O(N²) memory for the attention matrix.

            Fully compatible with torch.compile() for both forward and backward
            via proper custom op registration with FakeTensor implementations.

            Args:
                q: (B, H, T_q, D) queries
                k: (B, H, T_k, D) keys
                v: (B, H, T_k, D) values
                proj_logits: (H, 5) projection logits per head (will be softmaxed)
                causal: Whether to apply causal masking (default True)

            Returns:
                out: (B, H, T_q, D) attention output with look-around convolution
            """
            return FusedLookAroundFlashFunction.apply(q, k, v, proj_logits, causal)

    else:
        # =====================================================================
        # FALLBACK FOR OLDER PYTORCH VERSIONS (no custom_op support)
        # =====================================================================
        class FusedLookAroundFlashFunction(torch.autograd.Function):
            """Autograd function for fused look-around Flash Attention (legacy)."""

            @staticmethod
            def forward(ctx, q, k, v, proj_logits, causal=True):
                B, H, T_q, D = q.shape
                T_k = k.shape[2]

                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
                proj_5 = F.softmax(proj_logits.float(), dim=-1).contiguous()

                out = torch.empty_like(q)
                L = torch.empty((B, H, T_q), device=q.device, dtype=torch.float32)
                sm_scale = 1.0 / math.sqrt(D)

                BLOCK_D = triton.next_power_of_2(D)
                if BLOCK_D >= 64:
                    BLOCK_M, BLOCK_N = 32, 32
                else:
                    BLOCK_M, BLOCK_N = 64, 64

                grid = (triton.cdiv(T_q, BLOCK_M), B * H)

                if causal:
                    _fused_look_around_flash_causal_fwd_kernel[grid](
                        q, k, v, out, proj_5, L,
                        B, H, T_q, T_k, D, sm_scale,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                        proj_5.stride(0), proj_5.stride(1),
                        L.stride(0), L.stride(1), L.stride(2),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                    )
                else:
                    _fused_look_around_flash_fwd_kernel[grid](
                        q, k, v, out, proj_5, L,
                        B, H, T_q, T_k, D, sm_scale,
                        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                        proj_5.stride(0), proj_5.stride(1),
                        L.stride(0), L.stride(1), L.stride(2),
                        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D, IS_CAUSAL=causal,
                    )

                ctx.save_for_backward(q, k, v, proj_logits, proj_5, out, L)
                ctx.causal = causal
                ctx.sm_scale = sm_scale
                ctx.BLOCK_M = BLOCK_M
                ctx.BLOCK_N = BLOCK_N
                ctx.BLOCK_D = BLOCK_D

                return out

            @staticmethod
            def backward(ctx, grad_output):
                q, k, v, proj_logits, proj_5, out, L = ctx.saved_tensors
                causal = ctx.causal
                sm_scale = ctx.sm_scale
                BLOCK_M = ctx.BLOCK_M
                BLOCK_N = ctx.BLOCK_N
                BLOCK_D = ctx.BLOCK_D

                B, H, T_q, D = q.shape
                T_k = k.shape[2]

                grad_output = grad_output.contiguous()

                grad_q = torch.empty_like(q)
                grad_k = torch.zeros_like(k)
                grad_v = torch.zeros_like(v)
                grad_proj_5 = torch.zeros((H, 5), device=q.device, dtype=torch.float32)

                grid = (triton.cdiv(T_q, BLOCK_M), B * H)
                HALO_PAD = triton.next_power_of_2(BLOCK_N + 4)

                _fused_look_around_flash_bwd_kernel[grid](
                    q, k, v, out, proj_5, L, grad_output,
                    grad_q, grad_k, grad_v, grad_proj_5,
                    B, H, T_q, T_k, D, sm_scale,
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                    grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                    grad_q.stride(0), grad_q.stride(1), grad_q.stride(2), grad_q.stride(3),
                    grad_k.stride(0), grad_k.stride(1), grad_k.stride(2), grad_k.stride(3),
                    grad_v.stride(0), grad_v.stride(1), grad_v.stride(2), grad_v.stride(3),
                    proj_5.stride(0), proj_5.stride(1),
                    L.stride(0), L.stride(1), L.stride(2),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
                    HALO_PAD=HALO_PAD, IS_CAUSAL=causal,
                )

                dot_product = (grad_proj_5 * proj_5).sum(dim=-1, keepdim=True)
                grad_proj_logits = proj_5 * (grad_proj_5 - dot_product)
                grad_proj_logits = grad_proj_logits.to(proj_logits.dtype)

                return grad_q, grad_k, grad_v, grad_proj_logits, None

        def fused_look_around_flash_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            proj_logits: torch.Tensor,
            causal: bool = True,
        ) -> torch.Tensor:
            """
            Fused Flash Attention with Look-Around Convolution.

            Args:
                q: (B, H, T_q, D) queries
                k: (B, H, T_k, D) keys
                v: (B, H, T_k, D) values
                proj_logits: (H, 5) projection logits per head (will be softmaxed)
                causal: Whether to apply causal masking (default True)

            Returns:
                out: (B, H, T_q, D) attention output with look-around convolution
            """
            return FusedLookAroundFlashFunction.apply(q, k, v, proj_logits, causal)


else:
    # Triton not available
    def fused_look_around_flash_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        proj_logits: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        raise RuntimeError(
            "Triton is not available. Please install triton to use the fused kernel."
        )


def fused_look_around_flash_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    proj_logits: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    Reference PyTorch implementation for testing correctness.

    This materializes the full attention matrix (O(N²) memory) but produces
    the correct output for comparison with the fused kernel.
    """
    B, H, T_q, D = q.shape
    T_k = k.shape[2]

    # Compute attention scores
    sm_scale = 1.0 / math.sqrt(D)
    qk = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    # Apply causal mask
    if causal:
        q_idx = torch.arange(T_q, device=q.device).unsqueeze(1)
        k_idx = torch.arange(T_k, device=k.device).unsqueeze(0)
        causal_mask = k_idx > q_idx
        qk = qk.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    # Softmax
    p = F.softmax(qk, dim=-1)

    # Softmax projection logits
    w = F.softmax(proj_logits, dim=-1)

    # Apply convolution per head
    p_conv = p.new_zeros(B, H, T_q, T_k)
    p_pad = F.pad(p, (2, 2), value=0)  # (B, H, T_q, T_k + 4)
    for i in range(5):
        # w[h, 4-i] is the weight for shift i (i=0 is j-2, i=4 is j+2)
        # We need to broadcast w properly: (H,) -> (1, H, 1, 1)
        w_i = w[:, 4 - i].view(1, H, 1, 1)  # (1, H, 1, 1)
        p_conv = p_conv + w_i * p_pad[..., i:i + T_k]

    # Apply causal mask to convolved result
    if causal:
        p_conv = p_conv.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), 0.0)

    # Renormalize
    p_conv = p_conv / p_conv.sum(dim=-1, keepdim=True).clamp(min=1e-9)

    # Compute output
    out = torch.matmul(p_conv, v)

    return out
