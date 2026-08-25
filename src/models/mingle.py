from __future__ import annotations

import torch
from torch import nn

from src.models.layers import MingleLayer


class MingleModel(nn.Module):
    """Frozen Stage 3 MINGLE: MLP_1, L bipartite PMA layers, JK-CONCAT, MLP_CLS."""

    def __init__(
        self,
        node_input_dim: int = 832,
        semantic_dim: int = 768,
        hidden_dim: int = 48,
        num_heads: int = 4,
        num_layers: int = 2,
        num_classes: int = 25,
        num_real_hyperedges: int = 143946,
        expansion: int = 4,
        pairnorm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_real_hyperedges = num_real_hyperedges
        self.mlp1 = nn.Linear(semantic_dim, hidden_dim)
        layers = [
            MingleLayer(node_input_dim, hidden_dim, num_heads, expansion, pairnorm_eps)
        ]
        for _ in range(num_layers - 1):
            layers.append(MingleLayer(hidden_dim, hidden_dim, num_heads, expansion, pairnorm_eps))
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(hidden_dim * num_layers, num_classes)

    def forward(
        self,
        node_states: torch.Tensor,
        concept_semantics: torch.Tensor,
        note_semantics: torch.Tensor,
        incidence_node: torch.Tensor,
        incidence_edge: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        num_nodes = node_states.size(0)
        n_aug = torch.cat([note_semantics, concept_semantics], dim=0)
        h_e = self.mlp1(n_aug)
        num_hyperedges = h_e.size(0)

        layer_real_states = []
        a_e = None
        e_e = None
        x_v = node_states
        for layer in self.layers:
            x_v, e_e, a_e = layer(
                x_v,
                h_e,
                incidence_node,
                incidence_edge,
                num_nodes,
                num_hyperedges,
            )
            layer_real_states.append(e_e[: self.num_real_hyperedges])

        jk = torch.cat(layer_real_states, dim=-1)
        logits = self.classifier(jk)
        return {
            "logits": logits,
            "h_e": h_e,
            "a_e": a_e,
            "e_e": e_e,
            "x_v": x_v,
            "jk": jk,
            "n_aug": n_aug,
        }
