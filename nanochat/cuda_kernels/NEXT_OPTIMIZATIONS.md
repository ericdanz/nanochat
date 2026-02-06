# Flash Look-Around Attention: Next Optimizations

This document outlines the optimizations needed to improve performance on Blackwell (sm_120) GPUs.

## Current State (February 2026)

### Working Kernels
| Feature | Status | Notes |
|---------|--------|-------|
| SM90 WMMA kernel | ✅ Working | 16x16x16 tensor core ops, ~16x slower than SDPA |
| SM120 Pipelined kernel | ✅ Working | 2-stage cp.async, ~16x slower than SDPA |
| SM120 CuTe hybrid | ✅ Working | CuTe infrastructure with WMMA fallback |
| Online softmax + 5-tap halo | ✅ Done | Correct convolution with shared max |
| 64x64 tiling | ✅ Done | BLOCK_M=64, BLOCK_N=64 |
| GQA support | ✅ Done | Various Q:KV ratios tested |
| Tests passing | ✅ Done | All tests pass |

### In Progress
| Feature | Status | Issue |
|---------|--------|-------|
| SM120 CuTe TiledMMA | 🔄 Scaffolded | Uses WMMA fallback, needs tcgen05.mma |
| TMEM accumulator | 🔄 Scaffolded | Requires CUTE_ARCH_TCGEN05_TMEM_ENABLED |

### Broken/Incomplete
| Feature | Status | Issue |
|---------|--------|-------|
| SM120 F8 MMA kernel | ❌ Broken | Illegal memory access in P@V computation |
| Warp specialization | ❌ Not implemented | All warps do same work |

---

## Performance (Updated 2026-02-04)

RTX 5090, B=8, T=2048, 6 Q heads, 2 KV heads, D=64:

| Kernel | Time | vs SDPA |
|--------|------|---------|
| SDPA (no conv) | 0.17ms | 1x |
| SM90 WMMA | 2.72ms | ~16x |
| SM120 Pipelined | 3.05ms | ~18x |
| SM120 CuTe (WMMA fallback) | 3.79ms | ~22x |

The ~16-22x slowdown is expected because look-around computes 5x more attention
scores (for the 5-tap halo) and performs additional convolution.

---

## CuTe Hybrid Kernel Architecture

The new `flash_look_around_fwd_sm120_cute.cuh` provides a hybrid architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CuTe TiledMMA (QK^T)                         │
│    Q (SMEM) × K^T (SMEM) → S (TMEM)                            │
│    Uses tcgen05.mma, handles layout automatically               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                 TMEM → SMEM Copy (S scores)                     │
│    tcgen05.ld: TMEM → registers → SMEM                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│           Custom 5-Tap Convolution + Online Softmax             │
│    Our existing logic, but reading/writing SMEM                 │
│    - Find max across 5 shifts                                   │
│    - exp(S - max) weighted sum                                  │
│    - Update running max/sum for online softmax                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                    CuTe TiledMMA (P@V)                          │
│    P (SMEM) × V (SMEM) → O_partial (TMEM)                      │
│    Uses tcgen05.mma for 2x throughput                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                Rescale & Accumulate O                           │
│    Load O from TMEM, rescale with online softmax correction     │
└─────────────────────────────────────────────────────────────────┘
```

**Current Status:**
- Infrastructure in place for CuTe TiledMMA
- Currently uses WMMA as fallback (tcgen05 MMA not fully integrated)
- TMEM allocation scaffolded but disabled (requires CUTE_ARCH_TCGEN05_TMEM_ENABLED)
- Produces correct results (matches pipelined kernel)

---

## Optimization Roadmap

### Phase 1: Enable CuTe TiledMMA (High Priority)

**Current:** The CuTe kernel uses WMMA fallback.

**Required:**
1. Configure proper swizzled SMEM layouts using `UMMA::tile_to_mma_shape`
2. Enable `CUTE_ARCH_TCGEN05_TMEM_ENABLED` in build
3. Replace WMMA with `gemm(tiled_mma, tCrA, tCrB, tCtAcc)`
4. Add proper `umma_arrive`/`wait_barrier` synchronization

**Pattern from CUTLASS tutorial:**
```cpp
TiledMMA tiled_mma = make_tiled_mma(
    SM100_MMA_F16BF16_SS<cutlass::bfloat16_t, cutlass::bfloat16_t, float,
                        64, 64,  // MMA M, N
                        UMMA::Major::K, UMMA::Major::K>{}
);

// Single warp executes MMA
if (elect_one_warp) {
    gemm(tiled_mma, tCrA, tCrB, tCtAcc);
    cutlass::arch::umma_arrive(&mma_barrier);
}
wait_barrier(mma_barrier, phase_bit);
```

**Expected speedup:** 2-3x (larger MMA tiles, better scheduling)

---

### Phase 2: F8 MMA (Optional)

**Problem:** The original F8 MMA kernel has memory access issues.

**Status:** Deprioritized - CuTe approach is cleaner.

**If revisited:**
- Use CuTe's `SM100_MMA_F8_SS` atom
- Leverage automatic layout handling

---

### Phase 3: Warp Specialization

**Goal:** Split warps into producers (memory) + consumers (compute).

**Requires:** Working CuTe TiledMMA from Phase 1.

**Expected speedup:** 1.3-1.5x (cumulative: 3-5x)

---

## Key Files

| File | Purpose |
|------|---------|
| `flash_look_around_fwd_sm120_cute.cuh` | **NEW** CuTe hybrid kernel (WMMA fallback) |
| `flash_look_around_fwd_sm120_pipelined.cuh` | Working WMMA kernel (current best) |
| `flash_look_around_fwd_sm120.cuh` | Original SM120 kernel (broken F8 MMA) |
| `flash_look_around_kernel_sm120.h` | F8 MMA helpers, pipeline infrastructure |
| `flash_look_around_fwd_sm90.cuh` | SM90 kernel (WMMA-based) |
| `fused_look_around_flash.cu` | Dispatch logic, entry points |

---

## Reference Resources

- FlashAttention-3 paper: https://arxiv.org/abs/2407.08608
- CUTLASS SM100 examples: `cutlass/examples/cute/tutorial/blackwell/`
- FA3 implementation: https://github.com/Dao-AILab/flash-attention
- CuTe MMA tutorial: `cutlass/examples/cute/tutorial/blackwell/01_mma_sm100.cu`

---

## Build & Test

```bash
# Build SM120 kernel
export PATH=/usr/local/cuda-13.0/bin:$PATH
source /home/eric-danziger/dev/nanochat/.venv-cuda13/bin/activate
cd nanochat/cuda_kernels && python setup.py build_ext --inplace && cd ../..

# Run tests
export LD_LIBRARY_PATH=/home/eric-danziger/dev/nanochat/.venv-cuda13/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
python -m pytest tests/test_fused_look_around_flash_cuda.py -v

# Test CuTe kernel specifically
python -c "
import torch
import fused_look_around_flash_cuda as cuda_kernel

q = torch.randn(2, 6, 128, 64, dtype=torch.bfloat16, device='cuda')
k = torch.randn(2, 2, 128, 64, dtype=torch.bfloat16, device='cuda')
v = torch.randn(2, 2, 128, 64, dtype=torch.bfloat16, device='cuda')
w = torch.tensor([[0.1, 0.2, 0.4, 0.2, 0.1]] * 2, dtype=torch.float32, device='cuda')

out_cute, _ = cuda_kernel.forward_sm120_cute(q, k, v, w, True, -1)
out_pipe, _ = cuda_kernel.forward_sm120_pipelined(q, k, v, w, True, -1)
print(f'Max diff: {(out_cute.float() - out_pipe.float()).abs().max():.6f}')
"
```
