# FlashAttention-3 Optimizations TODO

## Current State (SM120 Kernel)

### Implemented and Working
- [x] F8 MMA using SM120_16x8x32_TN (E4M3 x E4M3 -> F32)
- [x] K=32 per instruction (2x throughput vs BF16 K=16)
- [x] Online softmax with 5-tap halo handling
- [x] Basic 64x64 tiling
- [x] WMMA fallback for non-F8 builds
- [x] Correct V loading for P@V (load_b_fragment_f8_direct)
- [x] SM120 kernel auto-dispatch enabled

### Infrastructure Exists But NOT Used
- [ ] TMA async loads - helpers exist in header (`cp_async`, `load_kv_async`) but kernel uses synchronous loads
- [ ] 3-stage pipeline - `SM120TensorStorage` has triple buffers but kernel is single-buffered
- [ ] Pipeline barriers - `mbarrier` primitives exist but unused

**Reason:** RTX 5090 has 99KB shared memory limit. Triple-buffering K/V would require:
- 3 × (80×64) × 2 bytes for K = 30KB
- 3 × (64×64) × 2 bytes for V = 24KB
- Plus Q, S, P, O_block, acc, F8 buffers... exceeds 99KB

Current single-buffer approach uses ~89KB and fits.

---

## Remaining Optimizations (Priority Order)

### 1. Reduce Shared Memory to Enable Pipelining (Highest Impact)

**Problem:** Can't use existing pipeline infrastructure due to 99KB limit.

**Solutions:**
a) **Smaller tiles:** Use 64x32 or 32x64 blocks instead of 64x64
b) **Eliminate redundant buffers:**
   - Don't store full P matrix, stream P@V computation
   - Fuse F8 quantization (don't need separate Q_e4m3, K_e4m3 buffers)
c) **Use 2-stage instead of 3-stage:** Reduces buffer size by 33%

**If we can fit 2-stage pipeline:**
```cpp
// Current: ~400 cycle stall per K/V block
for (k_block...) {
    load_K_sync();  // STALL
    load_V_sync();  // STALL
    compute();
}

// With 2-stage pipeline: overlap load and compute
load_K[0], V[0];  // Initial load
for (k_block...) {
    wait_for_stage(compute_stage);
    start_load_async(next_stage);  // Overlapped
    compute(compute_stage);
}
```

**Expected Speedup:** 1.5-2x

---

### 2. Warp Specialization (Medium Impact)

**Current:** All 4 warps do the same work in lockstep.

**FA3 Approach:** Split into producer (2 warps) and consumer (2 warps):
```
Producer Warps: Handle all memory operations
Consumer Warps: Handle all MMA compute

Timeline:
  Producers: [Load K0] [Load V0] [Load K1] [Load V1] ...
  Consumers:           [QK^T_0 ] [PV_0   ] [QK^T_1 ] ...
```

**Prerequisite:** Must have pipelining working first.

**Expected Speedup:** 1.3-1.5x (on top of pipelining)

---

### 3. Block-Scaled FP8 (Accuracy Improvement)

**Current Problem:** Static E4M3 range [-448, 448] loses precision for small values.

**Solution:** Dynamic per-block scaling:
```cpp
// Find block max
float block_max = block_reduce_max(abs(input));

// Scale to use full E4M3 range
float scale = 448.0f / block_max;
uint8_t quantized = float_to_e4m3(input * scale);

// After MMA, apply inverse scale
float result = mma_output * (scale_A * scale_B);
```

**Expected Impact:** Accuracy improvement, neutral on speed.

---

### 4. TMEM Accumulator (Blackwell-Specific, Low Priority)

**Current:** Accumulator uses 16KB of shared memory.

**Blackwell has 256KB TMEM per SM** that could store accumulator, freeing shared memory for larger K/V tiles.

**Complexity:** TMEM access is tightly coupled with tcgen05 instructions. Would need CUTLASS abstractions.

**Expected Speedup:** 1.1-1.2x (enables larger tiles)

---

## Quick Reference

### Build & Test
```bash
export PATH=/usr/local/cuda-13.0/bin:$PATH
source /home/eric-danziger/dev/nanochat/.venv-cuda13/bin/activate
cd nanochat/cuda_kernels && python setup.py build_ext --inplace && cd ../..

export LD_LIBRARY_PATH=/home/eric-danziger/dev/nanochat/.venv-cuda13/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${PWD}/nanochat/cuda_kernels:$PYTHONPATH"
python -m pytest tests/test_fused_look_around_flash_cuda.py -v
```

### Key Files
- `flash_look_around_fwd_sm120.cuh` - Main kernel (single-buffered, synchronous)
- `flash_look_around_kernel_sm120.h` - Helpers, pipeline infrastructure (unused)

### Current Performance Gap
~200x slower than SDPA (36ms vs 0.2ms for B=1, T=2048, D=64)

Target: 10-20x improvement with pipelining + warp specialization.
