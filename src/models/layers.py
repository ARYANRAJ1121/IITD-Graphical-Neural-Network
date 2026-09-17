from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from src.models.pairnorm import PairNorm
from src.models.pma import PMA


def _group_chunks(num_groups: int, nnz: int, dim: int, device: torch.device) -> list[tuple[int, int]]:
    """Split PMA by hyperedge/node groups so one 832-d gather fits in 16GB."""
    if device.type != "cuda" or nnz <= 0 or num_groups <= 0:
        return [(0, num_groups)]
    # source[nnz, dim] plus K/V (~2x) plus backward workspace. Stay well under 16GB.
    max_nnz = max(4096, int(4.0e7 / max(dim, 1)))
    if nnz <= max_nnz:
        return [(0, num_groups)]
    groups_per = max(256, int(num_groups * max_nnz / nnz))
    groups_per = min(groups_per, num_groups)
    return [(g0, min(g0 + groups_per, num_groups)) for g0 in range(0, num_groups, groups_per)]


class BipartiteSetBlock(nn.Module):
    """PMA + residual FFN + PairNorm on variable-size sets."""

    def __init__(self, dim: int, num_heads: int, expansion: int = 4, pairnorm_eps: float = 1e-5) -> None:
        super().__init__()
        self.pma = PMA(dim, num_heads)
        hidden = dim * expansion
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        self.pairnorm = PairNorm(eps=pairnorm_eps)

    def _pool_slice(
        self,
        table: torch.Tensor,
        row_index: torch.Tensor,
        group_index: torch.Tensor,
        g0: int,
        g1: int,
    ) -> torch.Tensor:
        mask = (group_index >= g0) & (group_index < g1)
        src = table[row_index[mask]]
        return self.pma.pool(src, group_index[mask] - g0, g1 - g0)

    def forward(
        self,
        table: torch.Tensor,
        row_index: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        chunks = _group_chunks(num_groups, int(row_index.numel()), table.size(-1), table.device)
        parts: list[torch.Tensor] = []
        use_ckpt = table.is_cuda and torch.is_grad_enabled() and len(chunks) > 1
        for g0, g1 in chunks:
            if use_ckpt:

                def _run(tbl, rows, groups, lo=g0, hi=g1):
                    return self._pool_slice(tbl, rows, groups, lo, hi)

                try:
                    part = checkpoint(_run, table, row_index, group_index, use_reentrant=False)
                except TypeError:
                    part = checkpoint(_run, table, row_index, group_index)
            else:
                part = self._pool_slice(table, row_index, group_index, g0, g1)
            parts.append(part)
        pooled = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return self.pairnorm(pooled + self.ffn(pooled))


class MingleLayer(nn.Module):
    """One MINGLE layer: f_V→E, MLP_2([A_e; H_e]), f_E→V."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int,
        num_heads: int,
        expansion: int = 4,
        pairnorm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.node_to_edge = BipartiteSetBlock(node_dim, num_heads, expansion, pairnorm_eps)
        self.mlp2 = nn.Linear(node_dim + hidden_dim, hidden_dim)
        self.edge_to_node = BipartiteSetBlock(hidden_dim, num_heads, expansion, pairnorm_eps)

    def forward(
        self,
        node_states: torch.Tensor,
        semantic_hyperedges: torch.Tensor,
        incidence_node: torch.Tensor,
        incidence_edge: torch.Tensor,
        num_nodes: int,
        num_hyperedges: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a_e = self.node_to_edge(node_states, incidence_node, incidence_edge, num_hyperedges)
        e_e = self.mlp2(torch.cat([a_e, semantic_hyperedges], dim=-1))
        x_v = self.edge_to_node(e_e, incidence_edge, incidence_node, num_nodes)
        return x_v, e_e, a_e
