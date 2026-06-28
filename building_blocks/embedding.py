import torch

import torch.nn as nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:

        return token_ids + 2
