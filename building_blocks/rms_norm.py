import torch
import torch.nn as nn

from einops import reduce, einsum  # pyright: ignore


class RMSNorm(nn.Module):

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_type = x.dtype
        x = x.to(torch.float32)
        mean_square: torch.Tensor = reduce(x**2, "... d -> ... 1", "mean")
        rms = torch.sqrt(mean_square + self.eps)
        normalised_x = x / rms * self.weight
        return normalised_x.to(dtype=in_type)


if __name__ == "__main__":
    rms_norm = RMSNorm(d_model=2)
    print(rms_norm.state_dict())
