from __future__ import annotations

import torch
from torch import nn

from src.models.pairnorm import PairNorm
from src.models.pma import PMA


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

    def forward(
        self,
        source: torch.Tensor,
        group_index: torch.Tensor,
        num_groups: int,
    ) -> torch.Tensor:
        pooled = self.pma.pool(source, group_index, num_groups)
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
        a_e = self.node_to_edge(node_states[incidence_node], incidence_edge, num_hyperedges)
        e_e = self.mlp2(torch.cat([a_e, semantic_hyperedges], dim=-1))
        x_v = self.edge_to_node(e_e[incidence_edge], incidence_node, num_nodes)
        return x_v, e_e, a_e
