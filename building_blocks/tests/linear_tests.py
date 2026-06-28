import pytest
import torch
from building_blocks.linear import Linear


def test_linear_initialization():
    """Test that Linear layer initializes with correct weight shape."""
    in_features = 10
    out_features = 5
    linear = Linear(in_features, out_features)

    assert linear.weights.shape == (in_features, out_features)
    assert isinstance(linear.weights, torch.nn.Parameter)


def test_linear_forward_shape():
    """Test that forward pass produces correct output shape."""
    in_features = 10
    out_features = 5
    linear = Linear(in_features, out_features)

    x = torch.randn(in_features)
    output = linear(x)

    assert output.shape == (out_features,)


def test_linear_forward_batched():
    """Test forward pass with batched input."""
    batch_size = 32
    in_features = 10
    out_features = 5
    linear = Linear(in_features, out_features)

    x = torch.randn(batch_size, in_features)
    output = linear(x)

    assert output.shape == (batch_size, out_features)


def test_linear_multi_dim_input():
    """Test forward pass with multi-dimensional input."""
    in_features = 10
    out_features = 5
    linear = Linear(in_features, out_features)

    # 3D input: (batch_size, seq_len, in_features)
    x = torch.randn(2, 8, in_features)
    output = linear(x)

    assert output.shape == (2, 8, out_features)


def test_linear_gradients():
    """Test that gradients flow through the layer."""
    in_features = 10
    out_features = 5
    linear = Linear(in_features, out_features)

    x = torch.randn(in_features, requires_grad=True)
    output = linear(x)
    loss = output.sum()
    loss.backward()

    # Check that weights have gradients
    assert linear.weights.grad is not None
    assert linear.weights.grad.shape == linear.weights.shape
    # Check that input has gradients
    assert x.grad is not None
