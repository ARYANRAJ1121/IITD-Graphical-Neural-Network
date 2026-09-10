import torch
from torch import nn


class PairNorm(nn.Module):
    """Center-and-scale PairNorm as frozen in equation_mapping.md §5."""

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_c = x - x.mean(dim=0, keepdim=True)
        scale = x_c.pow(2).sum(dim=-1).mean()
        return x_c / torch.sqrt(scale + self.eps)
