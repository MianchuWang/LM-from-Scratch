import torch
import torch.testing as testing
import pytest

from building_blocks.adapters import run_rmsnorm


def test_rmsnorm_ones_sanity():
    """Test 1: A tensor of all ones should return values slightly less than 1 (due to eps)."""
    d_model = 4
    x = torch.ones(1, d_model)
    w = torch.ones(d_model)

    out = run_rmsnorm(d_model, {"weight": w}, x)

    # RMS of ones is 1. Normalized output should be 1 / sqrt(1 + eps)
    expected_val = 1.0 / torch.sqrt(torch.tensor(1.0 + 1e-5))
    expected = torch.full((1, d_model), expected_val.item())

    testing.assert_close(out, expected)


def test_rmsnorm_zeros_eps():
    """Test 2: A tensor of all zeros tests the epsilon to ensure no division by zero."""
    d_model = 6
    x = torch.zeros(2, d_model)
    w = torch.ones(d_model)

    out = run_rmsnorm(d_model, {"weight": w}, x)

    # 0 / sqrt(0 + eps) should perfectly equal 0
    expected = torch.zeros(2, d_model)

    testing.assert_close(out, expected)


def test_rmsnorm_weight_scaling():
    """Test 3: Ensure the affine weight parameter scales the dimensions correctly."""
    d_model = 3
    x = torch.ones(1, d_model)
    w = torch.tensor([1.0, 2.0, 3.0])  # Distinct weights per channel

    out = run_rmsnorm(d_model, {"weight": w}, x)

    # Base RMS value without weights
    base_val = 1.0 / torch.sqrt(torch.tensor(1.0 + 1e-5))
    expected = torch.tensor([[1.0, 2.0, 3.0]]) * base_val

    testing.assert_close(out, expected)


def test_rmsnorm_multidimensional_broadcasting():
    """Test 4: Ensure it handles 3D tensors (Batch, Sequence, D_Model) properly."""
    batch, seq, d_model = 2, 5, 8
    x = torch.randn(batch, seq, d_model)
    w = torch.rand(d_model)

    out = run_rmsnorm(d_model, {"weight": w}, x)

    # The output shape must exactly match the input shape
    assert out.shape == (batch, seq, d_model)


def test_rmsnorm_mathematical_equivalence():
    """Test 5: Compare the einops implementation against a manual PyTorch calculation."""
    d_model = 16
    x = torch.randn(4, 10, d_model)
    w = torch.randn(d_model)
    eps = 1e-6

    # Get output from your einops implementation
    out_einops = run_rmsnorm(
        in_features=d_model, weights={"weight": w}, input=x, eps=eps
    )

    # Ground truth manual calculation (done in float32 to match your class logic)
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    out_manual = (x_fp32 * torch.rsqrt(variance + eps)).to(x.dtype) * w
    testing.assert_close(out_einops, out_manual)
