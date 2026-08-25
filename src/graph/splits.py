from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.data.stage3_bundle import Stage3Bundle


@dataclass
class SplitGraph:
    name: str
    original_index: np.ndarray
    notes: torch.Tensor
    labels: torch.Tensor
    pair_mask: torch.Tensor
    incidence_node: torch.Tensor
    incidence_edge: torch.Tensor
    num_real: int
    eval_offset: int


def patient_encounter_index(patient_ids: list[str], allowed: np.ndarray) -> np.ndarray:
    allowed_set = set(allowed.tolist())
    return np.array([i for i, patient in enumerate(patient_ids) if patient in allowed_set], dtype=np.int64)


def slice_incidence(
    incidence_node: torch.Tensor,
    incidence_edge: torch.Tensor,
    keep_real: np.ndarray,
    num_real_full: int,
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Keep listed real hyperedges (original ids) plus one self-loop per node."""
    remap = {int(old): new for new, old in enumerate(keep_real.tolist())}
    keep = set(remap)
    nodes: list[int] = []
    edges: list[int] = []
    for node, edge in zip(incidence_node.tolist(), incidence_edge.tolist(), strict=False):
        if edge >= num_real_full:
            continue
        if edge in keep:
            nodes.append(int(node))
            edges.append(remap[int(edge)])
    num_real = len(keep_real)
    for i in range(num_nodes):
        nodes.append(i)
        edges.append(num_real + i)
    return (
        torch.tensor(nodes, dtype=torch.long),
        torch.tensor(edges, dtype=torch.long),
        num_real,
    )


def build_split_graph(
    bundle: Stage3Bundle,
    encounter_index: np.ndarray,
    name: str,
    eval_offset: int = 0,
) -> SplitGraph:
    inc_n, inc_e, num_real = slice_incidence(
        bundle.incidence_node,
        bundle.incidence_edge,
        encounter_index,
        bundle.num_real,
        bundle.node_states.size(0),
    )
    idx = torch.from_numpy(encounter_index.astype(np.int64))
    return SplitGraph(
        name=name,
        original_index=encounter_index,
        notes=bundle.note_semantics[idx],
        labels=bundle.labels[idx],
        pair_mask=bundle.pair_mask[idx],
        incidence_node=inc_n,
        incidence_edge=inc_e,
        num_real=num_real,
        eval_offset=eval_offset,
    )


def build_eval_graph(bundle: Stage3Bundle, train_idx: np.ndarray, eval_idx: np.ndarray, name: str) -> SplitGraph:
    combined = np.concatenate([train_idx, eval_idx])
    graph = build_split_graph(bundle, combined, name, eval_offset=int(train_idx.size))
    return graph
