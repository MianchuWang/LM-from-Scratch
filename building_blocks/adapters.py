import torch

from typing import Dict

from linear import Linear
from embedding import Embedding


def run_linear(
    in_features: int,
    out_features: int,
    weights: Dict[str, torch.Tensor],
    input: torch.Tensor,
):
    model = Linear(in_features=in_features, out_features=out_features)
    model.load_state_dict({"weights": weights})
    return model(input)


def run_embedding(
    num_embeddings: int,
    embedding_dim: int,
    weights: Dict[str, torch.Tensor],
    token_ids: torch.Tensor,
):
    model = Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    model.load_state_dict({"weights": weights})
    return model(token_ids)
