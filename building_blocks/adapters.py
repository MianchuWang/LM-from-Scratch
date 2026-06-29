import torch

from typing import Dict

from building_blocks.linear import Linear
from building_blocks.embedding import Embedding
from building_blocks.rms_norm import RMSNorm
from building_blocks.swiglu import SwiGLU


def run_linear(
    in_features: int,
    out_features: int,
    weights: Dict[str, torch.Tensor],
    input: torch.Tensor,
):
    model = Linear(in_features=in_features, out_features=out_features)
    model.load_state_dict(weights)
    return model(input)


def run_embedding(
    num_embeddings: int,
    embedding_dim: int,
    weights: Dict[str, torch.Tensor],
    token_ids: torch.Tensor,
):
    model = Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    model.load_state_dict(weights)
    return model(token_ids)


def run_rmsnorm(
    in_features: int,
    weights: Dict[str, torch.Tensor],
    input: torch.Tensor,
    eps: float = 1e-5,
):
    normaliser = RMSNorm(in_features, eps)
    normaliser.load_state_dict(weights)
    return normaliser(input)


def run_swiglu(in_features: int, weights: Dict[str, torch.Tensor], input: torch.Tensor):
    model = SwiGLU(in_features)
    model.load_state_dict(weights)
    return model(input)
