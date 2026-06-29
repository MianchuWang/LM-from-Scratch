import pytest
import torch
import torch.testing as testing
import torch.nn.functional as F

from building_blocks.adapters import run_swiglu


def generate_swiglu_weights(in_features: int) -> dict[str, torch.Tensor]:
    """Helper function to generate valid state_dicts for SwiGLU."""
    out_features = round((8 / 3) * in_features / 64) * 64
    return {
        "linear1.weight": torch.randn(out_features, in_features),
        "linear2.weight": torch.randn(out_features, in_features),
        "linear3.weight": torch.randn(in_features, out_features),
    }


def test_swiglu_forward_shape():
    """Test 1: Test that forward pass produces correct output shape for a 1D input."""
    in_features = 64
    weights = generate_swiglu_weights(in_features)
    x = torch.randn(in_features)

    output = run_swiglu(in_features, weights, x)

    # SwiGLU projects back down to in_features
    assert output.shape == (in_features,)


def test_swiglu_forward_batched():
    """Test 2: Test forward pass with 2D batched input."""
    batch_size = 32
    in_features = 64
    weights = generate_swiglu_weights(in_features)
    x = torch.randn(batch_size, in_features)

    output = run_swiglu(in_features, weights, x)

    assert output.shape == (batch_size, in_features)


def test_swiglu_multi_dim_input():
    """Test 3: Test forward pass safely broadcasts over a 3D sequence input."""
    in_features = 64
    weights = generate_swiglu_weights(in_features)
    # 3D input: (batch_size, seq_len, in_features)
    x = torch.randn(2, 128, in_features)

    output = run_swiglu(in_features, weights, x)

    assert output.shape == (2, 128, in_features)


def test_swiglu_mathematical_equivalence():
    """Test 4: Verify the wrapper's output exactly matches the manual math."""
    in_features = 64
    weights = generate_swiglu_weights(in_features)

    # Scale down input to realistic neural network activation bounds
    x = torch.randn(in_features) * 0.1

    # Run through the adapter
    output = run_swiglu(in_features, weights, x)

    # Calculate expected output manually
    w1 = weights["linear1.weight"]
    w2 = weights["linear2.weight"]
    w3 = weights["linear3.weight"]

    # x @ W^T formula to match PyTorch standard linear memory layout
    gate_proj = F.silu(x @ w1.T)
    up_proj = x @ w2.T
    expected = (gate_proj * up_proj) @ w3.T

    # Allow a standard epsilon tolerance (1e-4) to account for floating-point
    # accumulation differences between the `einsum` and `@` C++ kernels.
    testing.assert_close(output, expected, rtol=1e-4, atol=1e-4)


def test_swiglu_gradients_flow():
    """Test 5: Ensure gradients pass through the wrapper properly."""
    in_features = 64
    weights = generate_swiglu_weights(in_features)

    # Track gradients on input
    x = torch.randn(in_features, requires_grad=True)

    output = run_swiglu(in_features, weights, x)

    # Output should require gradients
    assert output.requires_grad is True

    # Simulate backward pass
    loss = output.sum()
    loss.backward()

    # The gradient should flow all the way back to the input 'x'
    assert x.grad is not None
    assert x.grad.shape == x.shape
