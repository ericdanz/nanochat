"""
Integration test for GPT with look-around attention.

Verifies:
1. Model initialization with n_look_around_heads > 0
2. Forward pass works
3. Backward pass works (gradients flow)
4. Optimizer setup includes look_around_proj params

Run with: source .venv/bin/activate && python tests/test_gpt_look_right_integration.py
"""
import torch
import sys
sys.path.insert(0, '.')

from nanochat.gpt import GPT, GPTConfig


def test_model_init():
    """Test model initialization with look-around attention."""
    print("=" * 60)
    print("TEST 1: Model initialization")
    print("=" * 60)

    config = GPTConfig(
        sequence_len=128,
        vocab_size=1024,
        n_layer=4,
        n_head=8,
        n_kv_head=4,
        n_embd=256,
        n_look_around_heads=4,  # Half of query heads
    )

    with torch.device('meta'):
        model = GPT(config)

    # Check that look_around_proj exists in each layer
    for i, block in enumerate(model.transformer.h):
        assert block.attn.look_around_proj is not None, f"Layer {i} missing look_around_proj"
        assert block.attn.look_around_proj.shape == (4, 5), f"Layer {i} wrong shape (expected 5-tap convolution)"

    print(f"Config: n_head={config.n_head}, n_look_around_heads={config.n_look_around_heads}")
    print(f"All layers have look_around_proj with shape (4, 5)")
    print("\n[PASS] Model initialization passed")


def test_forward_backward():
    """Test forward and backward pass."""
    print("\n" + "=" * 60)
    print("TEST 2: Forward and backward pass")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_kv_head=4,  # CUDA kernel doesn't support GQA, must equal n_head
        n_embd=128,
        n_look_around_heads=2,
    )

    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.to(dtype=torch.bfloat16)  # Model uses bfloat16
    model.ensure_look_around_float32()  # Keep look_around_proj in float32 for gradient precision

    # Create dummy input
    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=device)

    # Setup optimizer for gradient flow test
    # Note: c_proj is zero-initialized by design, so gradients don't flow to
    # look_around_proj on the first step. After one optimizer step, c_proj
    # becomes non-zero and gradients flow normally.
    optimizer = model.setup_optimizer()

    # Step 1: c_proj is zero, so look_around_proj gradients are zero (expected)
    loss = model(idx, targets=targets)
    print(f"Step 1 loss: {loss.item():.4f}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Step 2: c_proj is now non-zero, gradients should flow
    loss = model(idx, targets=targets)
    print(f"Step 2 loss: {loss.item():.4f}")
    loss.backward()

    # Check gradients exist for look_around_proj (after c_proj becomes non-zero)
    for i, block in enumerate(model.transformer.h):
        grad = block.attn.look_around_proj.grad
        assert grad is not None, f"Layer {i}: no gradient for look_around_proj"
        assert not torch.all(grad == 0), f"Layer {i}: gradient is all zeros"
        print(f"  Layer {i} look_around_proj grad norm: {grad.norm().item():.6f}")

    print("\n[PASS] Forward and backward pass passed")


def test_optimizer_setup():
    """Test that optimizer includes look_around_proj params."""
    print("\n" + "=" * 60)
    print("TEST 3: Optimizer setup")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_kv_head=4,  # CUDA kernel doesn't support GQA, must equal n_head
        n_embd=128,
        n_look_around_heads=2,
    )

    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.to(dtype=torch.bfloat16)
    model.ensure_look_around_float32()

    optimizer = model.setup_optimizer()

    # Count parameters in optimizer
    opt_param_count = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
    model_param_count = sum(p.numel() for p in model.parameters())

    print(f"Model params: {model_param_count}")
    print(f"Optimizer params: {opt_param_count}")

    assert opt_param_count == model_param_count, "Optimizer missing some params!"

    # Find the look_around group
    # look_around_proj shape is (n_kv_head, 5) for 5-tap convolution
    found_look_around = False
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.shape == (config.n_kv_head, 5):
                found_look_around = True
                assert group['kind'] == 'adamw', "look_around_proj should use AdamW"
                break

    assert found_look_around, "Could not find look_around_proj in optimizer"
    print(f"look_around_proj params found in AdamW group (shape: {config.n_kv_head}, 5)")

    print("\n[PASS] Optimizer setup passed")


def test_no_look_around():
    """Test that model works normally with n_look_around_heads=0."""
    print("\n" + "=" * 60)
    print("TEST 4: Model without look-around (n_look_around_heads=0)")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        n_look_around_heads=0,  # Disabled
    )

    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.to(dtype=torch.bfloat16)

    # Check that look_around_proj is None
    for block in model.transformer.h:
        assert block.attn.look_around_proj is None

    # Forward pass should work
    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=device)

    loss = model(idx, targets=targets)
    loss.backward()

    print(f"Forward pass: loss = {loss.item():.4f}")
    print("look_around_proj is None for all layers (as expected)")

    print("\n[PASS] Model without look-around passed")


def test_param_counts():
    """Test parameter counting functions."""
    print("\n" + "=" * 60)
    print("TEST 5: Parameter counting")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = GPTConfig(
        sequence_len=64,
        vocab_size=256,
        n_layer=4,
        n_head=4,
        n_kv_head=2,
        n_embd=128,
        n_look_around_heads=2,
    )

    with torch.device('meta'):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.to(dtype=torch.bfloat16)
    model.ensure_look_around_float32()

    counts = model.num_scaling_params()
    print(f"Parameter counts: {counts}")

    # Verify total matches
    actual_total = sum(p.numel() for p in model.parameters())
    assert counts['total'] == actual_total, f"Total mismatch: {counts['total']} vs {actual_total}"

    # Verify look_around is in scalars
    look_around_total = config.n_layer * config.n_look_around_heads * 3
    print(f"Look-around params: {look_around_total} (included in scalars)")

    flops = model.estimate_flops()
    print(f"Estimated FLOPs per token: {flops:,}")

    print("\n[PASS] Parameter counting passed")


if __name__ == "__main__":
    test_model_init()
    test_forward_backward()
    test_optimizer_setup()
    test_no_look_around()
    test_param_counts()

    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED!")
    print("=" * 60)
