import torch
from building_blocks.embedding import Embedding


def test_embedding_initialization():
    """Test that the embedding layer initializes with the expected parameter shape."""
    embedding = Embedding(num_embeddings=4, embedding_dim=3)

    assert embedding.weight.shape == (4, 3)
    assert isinstance(embedding.weight, torch.nn.Parameter)


def test_embedding_forward_single_batch():
    """Test that forward returns one embedding vector per input token."""
    embedding = Embedding(num_embeddings=4, embedding_dim=3)
    token_ids = torch.tensor([2, 0])

    output = embedding(token_ids)

    assert output.shape == (2, 3)


def test_embedding_forward_batched_indices():
    """Test that forward supports batched token indices."""
    embedding = Embedding(num_embeddings=4, embedding_dim=3)
    token_ids = torch.tensor([[1, 3], [0, 2]])

    output = embedding(token_ids)

    assert output.shape == (2, 2, 3)


def test_embedding_returns_expected_rows():
    """Test that the layer returns the stored embedding rows for the requested IDs."""
    embedding = Embedding(num_embeddings=3, embedding_dim=2)
    embedding.weight.data.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))

    token_ids = torch.tensor([2, 0, 1])
    output = embedding(token_ids)

    expected = torch.tensor([[5.0, 6.0], [1.0, 2.0], [3.0, 4.0]])
    assert torch.allclose(output, expected)


def test_embedding_gradients_flow():
    """Test that gradients are computed for the embedding weights."""
    embedding = Embedding(num_embeddings=3, embedding_dim=2)
    token_ids = torch.tensor([1, 2], dtype=torch.long)

    output = embedding(token_ids)
    loss = output.sum()
    loss.backward()

    assert embedding.weight.grad is not None
    assert embedding.weight.grad.shape == embedding.weight.shape
