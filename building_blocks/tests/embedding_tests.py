import torch
import torch.testing as testing

from building_blocks.adapters import run_embedding


def test_embedding_output_shape():
    """Test 1: Check that the wrapper processes inputs and returns the correct feature dimension."""
    num_embeddings = 4
    embedding_dim = 3
    # Note: Passing a raw tensor to satisfy the wrapper's {"weight": weights} logic
    weights = {"weight": torch.randn(num_embeddings, embedding_dim)}
    token_ids = torch.tensor([0])

    output = run_embedding(num_embeddings, embedding_dim, weights, token_ids)

    # Output for a single token should be (1, embedding_dim)
    assert output.shape == (1, 3)


def test_embedding_forward_single_batch():
    """Test 2: Test that forward returns one embedding vector per input token in a 1D sequence."""
    num_embeddings = 4
    embedding_dim = 3
    weights = {"weight": torch.randn(num_embeddings, embedding_dim)}
    token_ids = torch.tensor([2, 0])

    output = run_embedding(num_embeddings, embedding_dim, weights, token_ids)

    # 2 tokens in, 2 vectors out
    assert output.shape == (2, 3)


def test_embedding_forward_batched_indices():
    """Test 3: Test that forward supports 2D batched token indices."""
    num_embeddings = 4
    embedding_dim = 3
    weights = {"weight": torch.randn(num_embeddings, embedding_dim)}
    # Shape: (Batch, Sequence Length) -> (2, 2)
    token_ids = torch.tensor([[1, 3], [0, 2]])

    output = run_embedding(num_embeddings, embedding_dim, weights, token_ids)

    # Shape should become (Batch, Sequence Length, Embedding Dim) -> (2, 2, 3)
    assert output.shape == (2, 2, 3)


def test_embedding_returns_expected_rows():
    """Test 4: Test that the layer returns the exact stored embedding rows for the requested IDs."""
    num_embeddings = 3
    embedding_dim = 2
    weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # ID 0  # ID 1  # ID 2
    weights = {"weight": weights}

    token_ids = torch.tensor([2, 0, 1])
    output = run_embedding(num_embeddings, embedding_dim, weights, token_ids)

    # The output should rearrange the rows based on the token_ids order
    expected = torch.tensor([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])

    testing.assert_close(output, expected)


def test_embedding_requires_grad():
    """Test 5: Test that the computational graph for gradients remains intact."""
    num_embeddings = 3
    embedding_dim = 2
    weights = {"weight": torch.randn(num_embeddings, embedding_dim)}
    token_ids = torch.tensor([1, 2], dtype=torch.long)

    output = run_embedding(num_embeddings, embedding_dim, weights, token_ids)

    # Because the internal Embedding.weight is an nn.Parameter,
    # the resulting output should automatically require gradients for backprop.
    assert output.requires_grad is True
