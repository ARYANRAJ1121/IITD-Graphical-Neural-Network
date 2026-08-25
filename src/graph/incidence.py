from __future__ import annotations

import torch


def build_augmented_incidence(
    node_ids: list[str],
    encounter_ids: list[str],
    node_encounter_pairs: list[tuple[str, str]],
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    Incidence of real visit hyperedges plus one self-loop hyperedge per node.
    Real hyperedges occupy [0, num_real); self-loops occupy [num_real, num_real + num_nodes).
    Returns (incidence_node, incidence_edge, num_real, num_total_hyperedges).
    """
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}
    encounter_index = {encounter_id: i for i, encounter_id in enumerate(encounter_ids)}
    num_nodes = len(node_ids)
    num_real = len(encounter_ids)

    nodes: list[int] = []
    edges: list[int] = []
    for node_id, encounter_id in node_encounter_pairs:
        nodes.append(node_index[node_id])
        edges.append(encounter_index[encounter_id])

    for i in range(num_nodes):
        nodes.append(i)
        edges.append(num_real + i)

    incidence_node = torch.tensor(nodes, dtype=torch.long)
    incidence_edge = torch.tensor(edges, dtype=torch.long)
    return incidence_node, incidence_edge, num_real, num_real + num_nodes
