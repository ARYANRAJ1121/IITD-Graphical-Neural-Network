from __future__ import annotations

import math

import torch
from torch import nn


def grouped_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Softmax of `scores` within groups defined by `index` in [0, dim_size)."""
    if scores.numel() == 0:
        return scores
    min_value = torch.finfo(scores.dtype).min
    grouped_max = torch.full((dim_size,), min_value, device=scores.device, dtype=scores.dtype)
    grouped_max = grouped_max.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
    shifted = torch.exp(scores - grouped_max[index])
    grouped_sum = torch.zeros(dim_size, device=scores.device, dtype=scores.dtype)
    grouped_sum.scatter_add_(0, index, shifted)
    return shifted / grouped_sum[index].clamp_min(1e-12)


class PMA(nn.Module):
    """Pooling by Multihead Attention with learned queries W_i^Q. Not nn.MultiheadAttention."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.key_weight = nn.Parameter(torch.empty(num_heads, dim, self.head_dim))
        self.value_weight = nn.Parameter(torch.empty(num_heads, dim, self.head_dim))
        self.output_weight = nn.Parameter(torch.empty(dim, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query)
        nn.init.xavier_uniform_(self.key_weight)
        nn.init.xavier_uniform_(self.value_weight)
        nn.init.xavier_uniform_(self.output_weight)

    def pool(
        self,
        source: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        """
        source: [nnz, dim] stacked set elements
        group_index: [nnz] group id in [0, num_groups)
        returns: [num_groups, dim] (zeros for empty groups)
        """
        if source.numel() == 0:
            return source.new_zeros(num_groups, self.dim)

        keys = torch.einsum("nd,hdk->nhk", source, self.key_weight)
        values = torch.einsum("nd,hdk->nhk", source, self.value_weight)
        scale = math.sqrt(self.head_dim)
        scores = (keys * self.query.unsqueeze(0)).sum(dim=-1) / scale

        head_outputs = []
        for head in range(self.num_heads):
            alpha = grouped_softmax(scores[:, head], group_index, num_groups)
            weighted = values[:, head, :] * alpha.unsqueeze(-1)
            pooled = source.new_zeros(num_groups, self.head_dim)
            pooled.scatter_add_(0, group_index.unsqueeze(-1).expand_as(weighted), weighted)
            head_outputs.append(pooled)

        concatenated = torch.cat(head_outputs, dim=-1)
        return concatenated @ self.output_weight
