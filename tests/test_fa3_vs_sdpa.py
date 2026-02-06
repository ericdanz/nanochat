"""Compare FA3 local build vs torch SDPA for correctness and performance."""
import torch
import torch.nn.functional as F
import time


def sdpa_forward(q, k, v, causal=True, window_size=(-1, -1)):
    """SDPA reference: (B, T, H, D) -> transpose -> SDPA -> transpose back."""
    q_s = q.transpose(1, 2)
    k_s = k.transpose(1, 2)
    v_s = v.transpose(1, 2)
    enable_gqa = q_s.size(1) != k_s.size(1)
    y = F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=causal, enable_gqa=enable_gqa)
    return y.transpose(1, 2)


def sdpa_forward_backward(q, k, v, causal=True):
    """SDPA forward+backward."""
    out = sdpa_forward(q, k, v, causal=causal)
    loss = out.sum()
    loss.backward()
    return out


def fa3_forward_backward(fa3, q, k, v, causal=True):
    """FA3 forward+backward."""
    out = fa3.flash_attn_func(q, k, v, causal=causal)
    loss = out.sum()
    loss.backward()
    return out


def benchmark(fn, warmup=10, iters=50):
    """Benchmark with CUDA events for accurate timing."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms per iteration


def check_close(name, a, b, atol=1e-2, rtol=1e-2):
    """Check tensors are close, print stats."""
    diff = (a.float() - b.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    ok = torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  {name}: {status}  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    return ok


def test_correctness(fa3):
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    all_pass = True

    # Test 1: Forward, standard shapes
    print("\n--- Forward (B=2, T=512, H=16, D=128, causal) ---")
    B, T, H, D = 2, 512, 16, 128
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")

    out_sdpa = sdpa_forward(q, k, v, causal=True)
    out_fa3 = fa3.flash_attn_func(q, k, v, causal=True)
    all_pass &= check_close("output", out_fa3, out_sdpa)

    # Test 2: Forward, GQA (fewer KV heads)
    print("\n--- Forward GQA (B=2, T=512, Hq=16, Hkv=4, D=128, causal) ---")
    H_kv = 4
    torch.manual_seed(123)
    q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(B, T, H_kv, D, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(B, T, H_kv, D, dtype=torch.bfloat16, device="cuda")

    out_sdpa = sdpa_forward(q, k, v, causal=True)
    out_fa3 = fa3.flash_attn_func(q, k, v, causal=True)
    all_pass &= check_close("output", out_fa3, out_sdpa)

    # Test 3: Forward+Backward
    print("\n--- Forward+Backward (B=2, T=256, H=8, D=128, causal) ---")
    B, T, H, D = 2, 256, 8, 128
    torch.manual_seed(99)
    q_base = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    k_base = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    v_base = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")

    # SDPA backward
    q1, k1, v1 = q_base.clone().requires_grad_(), k_base.clone().requires_grad_(), v_base.clone().requires_grad_()
    out_sdpa = sdpa_forward(q1, k1, v1, causal=True)
    out_sdpa.sum().backward()

    # FA3 backward
    q2, k2, v2 = q_base.clone().requires_grad_(), k_base.clone().requires_grad_(), v_base.clone().requires_grad_()
    out_fa3 = fa3.flash_attn_func(q2, k2, v2, causal=True)
    out_fa3.sum().backward()

    all_pass &= check_close("fwd output", out_fa3, out_sdpa)
    all_pass &= check_close("dq", q2.grad, q1.grad, atol=2e-2, rtol=2e-2)
    all_pass &= check_close("dk", k2.grad, k1.grad, atol=2e-2, rtol=2e-2)
    all_pass &= check_close("dv", v2.grad, v1.grad, atol=2e-2, rtol=2e-2)

    # Test 4: KV cache
    print("\n--- KV Cache (B=4, T_new=1, H=16, Hkv=4, D=128) ---")
    B, T_new, H, D, H_kv, T_max = 4, 1, 16, 128, 4, 2048
    torch.manual_seed(77)
    q = torch.randn(B, T_new, H, D, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(B, T_max, H_kv, D, dtype=torch.bfloat16, device="cuda")
    v_cache = torch.randn(B, T_max, H_kv, D, dtype=torch.bfloat16, device="cuda")
    k_new = torch.randn(B, T_new, H_kv, D, dtype=torch.bfloat16, device="cuda")
    v_new = torch.randn(B, T_new, H_kv, D, dtype=torch.bfloat16, device="cuda")
    cache_seqlens = torch.tensor([512, 1024, 256, 768], dtype=torch.int32, device="cuda")

    # FA3 kvcache
    k_cache_fa3 = k_cache.clone()
    v_cache_fa3 = v_cache.clone()
    out_fa3 = fa3.flash_attn_with_kvcache(
        q, k_cache_fa3, v_cache_fa3, k=k_new, v=v_new,
        cache_seqlens=cache_seqlens, causal=True,
    )

    # SDPA reference: manually update cache and run attention
    k_cache_ref = k_cache.clone()
    v_cache_ref = v_cache.clone()
    for b in range(B):
        pos = cache_seqlens[b].item()
        k_cache_ref[b, pos : pos + T_new] = k_new[b]
        v_cache_ref[b, pos : pos + T_new] = v_new[b]

    # Build full K, V up to pos+T_new for each batch element, pad to same length
    max_len = (cache_seqlens + T_new).max().item()
    k_full = torch.zeros(B, max_len, H_kv, D, dtype=torch.bfloat16, device="cuda")
    v_full = torch.zeros(B, max_len, H_kv, D, dtype=torch.bfloat16, device="cuda")
    for b in range(B):
        end = cache_seqlens[b].item() + T_new
        k_full[b, :end] = k_cache_ref[b, :end]
        v_full[b, :end] = v_cache_ref[b, :end]

    # SDPA attention (single query token against full cache)
    q_s = q.transpose(1, 2)  # (B, H, 1, D)
    k_s = k_full.transpose(1, 2)  # (B, H_kv, max_len, D)
    v_s = v_full.transpose(1, 2)
    # Build causal mask: query at position cache_seqlens, keys at 0..max_len-1
    mask = torch.zeros(B, 1, T_new, max_len, dtype=torch.bool, device="cuda")
    for b in range(B):
        end = cache_seqlens[b].item() + T_new
        mask[b, :, :, :end] = True
    out_sdpa = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=mask, enable_gqa=True)
    out_sdpa = out_sdpa.transpose(1, 2)

    all_pass &= check_close("kvcache output", out_fa3, out_sdpa)

    print("\n" + "=" * 60)
    print(f"CORRECTNESS: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print("=" * 60)
    return all_pass


def test_performance(fa3):
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARKS")
    print("=" * 60)

    configs = [
        # (B, T, H, H_kv, D, label)
        (2, 1024, 16, 16, 128, "Training B=2 T=1024 H=16"),
        (4, 2048, 16, 16, 128, "Training B=4 T=2048 H=16"),
        (8, 4096, 16, 16, 128, "Training B=8 T=4096 H=16"),
        (2, 1024, 16, 4, 128, "Training GQA B=2 T=1024 H=16 Hkv=4"),
    ]

    print(f"\n{'Config':<45} {'SDPA (ms)':>10} {'FA3 (ms)':>10} {'Speedup':>10}")
    print("-" * 80)

    for B, T, H, H_kv, D, label in configs:
        torch.manual_seed(0)
        q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, T, H_kv, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, T, H_kv, D, dtype=torch.bfloat16, device="cuda")

        # Forward only
        t_sdpa = benchmark(lambda: sdpa_forward(q, k, v, causal=True))
        t_fa3 = benchmark(lambda: fa3.flash_attn_func(q, k, v, causal=True))
        speedup = t_sdpa / t_fa3
        print(f"  Fwd  {label:<40} {t_sdpa:>9.3f}  {t_fa3:>9.3f}  {speedup:>9.2f}x")

        # Forward+Backward
        def run_sdpa_fwdbwd():
            q_, k_, v_ = q.detach().requires_grad_(), k.detach().requires_grad_(), v.detach().requires_grad_()
            sdpa_forward_backward(q_, k_, v_, causal=True)

        def run_fa3_fwdbwd():
            q_, k_, v_ = q.detach().requires_grad_(), k.detach().requires_grad_(), v.detach().requires_grad_()
            fa3_forward_backward(fa3, q_, k_, v_, causal=True)

        t_sdpa = benchmark(run_sdpa_fwdbwd)
        t_fa3 = benchmark(run_fa3_fwdbwd)
        speedup = t_sdpa / t_fa3
        print(f"  Fwd+Bwd  {label:<36} {t_sdpa:>9.3f}  {t_fa3:>9.3f}  {speedup:>9.2f}x")

    # KV cache benchmark
    print(f"\n{'KV Cache Config':<45} {'SDPA (ms)':>10} {'FA3 (ms)':>10} {'Speedup':>10}")
    print("-" * 80)

    kv_configs = [
        (1, 1, 16, 4, 128, 2048, "Decode B=1 Tctx=2048 H=16 Hkv=4"),
        (8, 1, 16, 4, 128, 2048, "Decode B=8 Tctx=2048 H=16 Hkv=4"),
        (32, 1, 16, 4, 128, 2048, "Decode B=32 Tctx=2048 H=16 Hkv=4"),
        (1, 1, 16, 4, 128, 8192, "Decode B=1 Tctx=8192 H=16 Hkv=4"),
        (8, 1, 16, 4, 128, 8192, "Decode B=8 Tctx=8192 H=16 Hkv=4"),
    ]

    for B, T_new, H, H_kv, D, T_ctx, label in kv_configs:
        T_max = T_ctx + 64
        torch.manual_seed(0)
        q = torch.randn(B, T_new, H, D, dtype=torch.bfloat16, device="cuda")
        k_cache = torch.randn(B, T_max, H_kv, D, dtype=torch.bfloat16, device="cuda")
        v_cache = torch.randn(B, T_max, H_kv, D, dtype=torch.bfloat16, device="cuda")
        k_new = torch.randn(B, T_new, H_kv, D, dtype=torch.bfloat16, device="cuda")
        v_new = torch.randn(B, T_new, H_kv, D, dtype=torch.bfloat16, device="cuda")
        cache_seqlens = torch.full((B,), T_ctx, dtype=torch.int32, device="cuda")

        def run_fa3_kv():
            kc, vc = k_cache.clone(), v_cache.clone()
            fa3.flash_attn_with_kvcache(q, kc, vc, k=k_new, v=v_new, cache_seqlens=cache_seqlens, causal=True)

        def run_sdpa_kv():
            # Manually update cache + SDPA
            kc, vc = k_cache.clone(), v_cache.clone()
            for b in range(B):
                pos = cache_seqlens[b].item()
                kc[b, pos:pos+T_new] = k_new[b]
                vc[b, pos:pos+T_new] = v_new[b]
            end = T_ctx + T_new
            q_s = q.transpose(1, 2)
            k_s = kc[:, :end].transpose(1, 2)
            v_s = vc[:, :end].transpose(1, 2)
            F.scaled_dot_product_attention(q_s, k_s, v_s, is_causal=False, enable_gqa=True)

        t_fa3 = benchmark(run_fa3_kv)
        t_sdpa = benchmark(run_sdpa_kv)
        speedup = t_sdpa / t_fa3
        print(f"  {label:<43} {t_sdpa:>9.3f}  {t_fa3:>9.3f}  {speedup:>9.2f}x")


if __name__ == "__main__":
    import flash_attn_3.flash_attn_interface as fa3
    print(f"torch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    test_correctness(fa3)
    test_performance(fa3)
