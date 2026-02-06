# Fused Look-Around Flash Attention CUDA Kernel

## Motivation

Standard attention computes a weighted sum over values where weights come from softmax of query-key similarity. The highest-weighted position gets the most influence on the output.

**Look-around attention** allows select heads to not just attend to the highest-matching token, but also to nearby tokens. This is implemented via a learnable 5-tap convolution on attention probabilities, letting the model "look around" the peak attention position.

For example, if token 500 attends most strongly to token 347, look-around attention can also incorporate information from tokens 345, 346, 348, 349 with learnable weights.

## Mathematical Formulation

### Standard Attention
```
S = Q @ K.T / sqrt(d)      # Scores: (T_q, T_k)
P = softmax(S, dim=-1)     # Probabilities: (T_q, T_k), each row sums to 1
O = P @ V                  # Output: (T_q, D)
```

### Look-Around Attention
```
S = Q @ K.T / sqrt(d)      # Scores: (T_q, T_k)
P = softmax(S, dim=-1)     # Probabilities: (T_q, T_k)

# 5-tap convolution on probabilities (NOT scores)
# For each output position j (which V to weight):
P_conv[j] = w0 * P[j+2] + w1 * P[j+1] + w2 * P[j] + w3 * P[j-1] + w4 * P[j-2]

# Where w = softmax(learnable_logits) ensures weights sum to 1
# w[2] is the "center" weight - when w = [0,0,1,0,0], this reduces to standard attention

O = P_conv @ V             # Output weighted by convolved probabilities
```

### Key Insight: Order of Operations

**We convolve AFTER softmax, not before.** This is critical because:

1. Softmax probabilities represent "how much to attend to each position"
2. Convolution spreads this attention to neighboring positions
3. If we convolved scores before softmax, the semantics would be different (mixing raw similarities rather than attention weights)

## Causality Constraint

For autoregressive (causal) language modeling, position `i` must not see any information from positions `j > i`. This creates two constraints:

### 1. Score Masking (Standard)
```
if k_pos > q_pos:
    S[q_pos, k_pos] = -inf  # Masked before softmax
```

### 2. V Position Masking (Look-Around Specific)
Even after convolving, we must ensure no future V contributes to output:
```
if v_pos > q_pos:
    P_conv[q_pos, v_pos] = 0  # No contribution from future V
```

**Why is this needed?** Consider query at position 5 attending to key at position 4:
- The convolution `P_conv[4] = w0*P[6] + w1*P[5] + w2*P[4] + ...` would use P[6]
- P[6] involves K[6] which is masked (-inf), so P[6] = 0 after softmax
- But if it weren't zero, V[4] would receive weight from P[6], leaking K[6] information
- More importantly, after convolution, P_conv[6] could be non-zero from P[4], P[5]
- This would weight V[6] which is a future value - MUST be masked to 0

## Kernel Implementation

### Flash Attention Style
The kernel uses online softmax (Flash Attention algorithm) to avoid materializing the full (T_q, T_k) attention matrix:

1. Process K/V in blocks of size BLOCK_N
2. Maintain running max `m` and sum `l` for stable softmax
3. Rescale accumulated output when max changes

### Halo Loading for Convolution
Since we need P[j-2] through P[j+2] for each output position j, we load K with a "halo":
```
For K block [k_start, k_start + BLOCK_N):
  Load K positions [k_start - 2, k_start + BLOCK_N + 2)  # 4 extra positions
  Compute scores for all halo positions
  Convolve within the block
```

### Shared Normalization
All 5 shifted probability values share the same softmax denominator:
```
# Find global max across ALL shifts for numerical stability
m_new = max(S[j-2], S[j-1], S[j], S[j+1], S[j+2]) for all j

# Compute exp with shared max
p_j = exp(S[j] - m_new)  # All use same m_new

# Convolve
p_conv = w0*p[j+2] + w1*p[j+1] + w2*p[j] + w3*p[j-1] + w4*p[j-2]

# Accumulate into running sum
l += p_conv
```

This is mathematically equivalent to:
1. Computing full softmax P = softmax(S)
2. Convolving P
3. Renormalizing

But more numerically stable and memory efficient.

## Parameters

- **proj_logits**: `(H, 5)` learnable parameters per head
  - Passed through softmax to get weights `w = [w0, w1, w2, w3, w4]`
  - Initialized near identity: `[-2, -2, 2, -2, -2]` → softmax ≈ `[0.02, 0.02, 0.92, 0.02, 0.02]`
  - Allows gradient flow while starting close to standard attention

- **window_left**: Sliding window attention support
  - `-1`: Full attention (attend to all previous positions)
  - `>= 0`: Only attend to positions within `[q_pos - window_left, q_pos]`

## Memory Layout

```
Inputs:
  Q: (B, H, T_q, D) bfloat16
  K: (B, H, T_k, D) bfloat16
  V: (B, H, T_k, D) bfloat16
  proj_weights: (H, 5) float32 (pre-softmaxed)

Outputs:
  O: (B, H, T_q, D) bfloat16
  LSE: (B, H, T_q) float32 (log-sum-exp for backward)
```

## Limitations

1. **No GQA support**: Requires `n_kv_head == n_head` (Q and K must have same head count)
2. **Head dimension**: Optimized for D=64 and D=128
3. **Sequence length**: Must be divisible by block size for optimal performance

## Backward Pass

The backward pass computes gradients for Q, K, V, and proj_weights:

1. **dV**: Similar to standard attention backward
2. **dK, dQ**: Requires transposed convolution (reversing the shift directions)
3. **d_proj_weights**: Sum of `P * dP_conv` contributions

Key insight for transposed convolution:
```
Forward:  P_conv[j] = w0*P[j+2] + w1*P[j+1] + w2*P[j] + w3*P[j-1] + w4*P[j-2]
Backward: dP[j] = w0*dP_conv[j-2] + w1*dP_conv[j-1] + w2*dP_conv[j] + w3*dP_conv[j+1] + w4*dP_conv[j+2]
```

## Files

- `fused_look_around_fwd.cuh`: Forward kernel
- `fused_look_around_bwd.cuh`: Backward kernel
- `fused_look_around_flash.cu`: Entry points
- `bindings.cpp`: Python bindings
- `__init__.py`: PyTorch autograd wrapper
