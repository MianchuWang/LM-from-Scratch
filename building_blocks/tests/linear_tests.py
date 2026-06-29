import pytest
import torch
import torch.testing as testing

from building_blocks.adapters import run_linear


def test_linear_forward_shape():
    """Test 1: Test that forward pass produces correct output shape for a 1D input."""
    in_features = 10
    out_features = 5
    # Passing a raw tensor to satisfy the wrapper's {"weight": weights} logic
    weights = torch.randn(in_features, out_features)
    weights = {"weight": weights}
    x = torch.randn(in_features)
    output = run_linear(in_features, out_features, weights, x)

    assert output.shape == (out_features,)


def test_linear_forward_batched():
    """Test 2: Test forward pass with 2D batched input."""
    batch_size = 32
    in_features = 10
    out_features = 5
    weights = torch.randn(in_features, out_features)
    weights = {"weight": weights}
    x = torch.randn(batch_size, in_features)
    output = run_linear(in_features, out_features, weights, x)

    assert output.shape == (batch_size, out_features)


def test_linear_multi_dim_input():
    """Test 3: Test forward pass safely broadcasts over a 3D input."""
    in_features = 10
    out_features = 5
    weights = torch.randn(in_features, out_features)
    weights = {"weight": weights}
    # 3D input: (batch_size, seq_len, in_features)
    x = torch.randn(2, 8, in_features)
    output = run_linear(in_features, out_features, weights, x)

    assert output.shape == (2, 8, out_features)


def test_linear_mathematical_correctness():
    """Test 4: Verify the matrix multiplication computes exact expected values."""
    in_features = 3
    out_features = 2

    # Custom weights: 3 rows (in), 2 cols (out)
    weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    weights = {"weight": weights}
    # Input vector of ones
    x = torch.tensor([1.0, 1.0, 1.0])

    output = run_linear(in_features, out_features, weights, x)

    # Dot product calculation:
    # out_0 = (1*1) + (1*3) + (1*5) = 9.0
    # out_1 = (1*2) + (1*4) + (1*6) = 12.0
    expected = torch.tensor([9.0, 12.0])

    testing.assert_close(output, expected)


def test_linear_gradients_flow():
    """Test 5: Test that gradients flow properly through the wrapper output to the input."""
    in_features = 10
    out_features = 5
    weights = torch.randn(in_features, out_features)
    weights = {"weight": weights}

    # We need to track gradients on the input tensor
    x = torch.randn(in_features, requires_grad=True)

    output = run_linear(in_features, out_features, weights, x)

    # The output should remain connected to the internal parameter's graph
    assert output.requires_grad is True

    # Simulate a backward pass
    loss = output.sum()
    loss.backward()

    # We can't directly check model.weight.grad because the model is discarded,
    # but we CAN check that the gradient successfully flowed back to our input 'x'.
    assert x.grad is not None
    assert x.grad.shape == x.shape
