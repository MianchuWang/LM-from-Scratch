import torch
import torch.nn as nn

from building_blocks.linear import Linear


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        out_features = round((8 / 3) * in_features / 64) * 64
        self.linear1 = Linear(
            in_features=in_features,
            out_features=out_features,
            dtype=dtype,
            device=device,
        )
        self.linear2 = Linear(
            in_features=in_features,
            out_features=out_features,
            dtype=dtype,
            device=device,
        )
        self.linear3 = Linear(
            in_features=out_features,
            out_features=in_features,
            dtype=dtype,
            device=device,
        )

    # SiLU(x) = x * sigma(x)
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    # GLU(x, W_1, W_2) = sigma(x W_1) * x W_2
    def glu(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear1(x)) * self.linear2(x)

    # SwiGLU W_3 (SiLU(x W_1) * x W_2)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear3(self.silu(self.linear1(x)) * self.linear2(x))
