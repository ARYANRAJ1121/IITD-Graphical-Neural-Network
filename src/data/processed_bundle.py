from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.next_visit_labels import FROZEN_TOP25, FROZEN_TOP25_CODES
from src.graph.incidence import build_augmented_incidence


@dataclass
class ProcessedBundle:
    node_ids: list[str]
    encounter_ids: list[str]
    patient_ids: list[str]
    node_states: torch.Tensor
    concept_semantics: torch.Tensor
    note_semantics: torch.Tensor
    incidence_node: torch.Tensor
    incidence_edge: torch.Tensor
    num_real: int
    num_self_loops: int
    labels: torch.Tensor
    pair_mask: torch.Tensor
    split_ids: dict[str, np.ndarray]
    source: str


def _code_suffix(node_id: str) -> str:
    return str(node_id).rsplit("|", 1)[-1]


def build_next_visit_labels(
    encounters: pd.DataFrame,
    edges: pd.DataFrame,
    node_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """y[t] = top-25 membership of visit t+1; pair_mask[t]=False for last visits."""
    node_lookup = {node_id: i for i, node_id in enumerate(node_ids)}
    target_cols = []
    for code in FROZEN_TOP25_CODES:
        matches = [node_id for node_id in node_ids if _code_suffix(node_id) == code]
        target_cols.append(node_lookup[matches[0]] if matches else None)

    enc_index = {eid: i for i, eid in enumerate(encounters["encounter_id"].tolist())}
    membership = [set() for _ in range(len(encounters))]
    for _, row in edges.iterrows():
        enc_i = enc_index.get(row["encounter_id"])
        node_i = node_lookup.get(row["node_id"])
        if enc_i is not None and node_i is not None:
            membership[enc_i].add(node_i)

    patients = encounters["patient_id"].tolist()
    n = len(encounters)
    labels = np.zeros((n, 25), dtype=np.float32)
    pair_mask = np.zeros(n, dtype=bool)
    for i in range(n - 1):
        if patients[i] != patients[i + 1]:
            continue
        pair_mask[i] = True
        next_nodes = membership[i + 1]
        for class_i, node_i in enumerate(target_cols):
            if node_i is not None and node_i in next_nodes:
                labels[i, class_i] = 1.0
    return labels, pair_mask


def split_patients(patient_ids: list[str], ratios: tuple[float, float, float], seed: int) -> dict[str, np.ndarray]:
    unique = np.array(sorted(set(patient_ids)))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = unique[:n_train]
    val = unique[n_train : n_train + n_val]
    test = unique[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def example_index(patient_ids: list[str], pair_mask: np.ndarray, split_ids: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    patients = np.array(patient_ids)
    out = {}
    for name, ids in split_ids.items():
        allowed = set(ids.tolist())
        out[name] = np.where(pair_mask & np.array([p in allowed for p in patients]))[0]
    return out


def load_processed_bundle(processed_dir: Path, seed: int, ratios: tuple[float, float, float]) -> ProcessedBundle | None:
    required = [
        processed_dir / "concept_nodes.csv",
        processed_dir / "encounter_hyperedges.csv",
        processed_dir / "node_encounter_edges.csv",
        processed_dir / "node_embeddings.npy",
        processed_dir / "note_embeddings.npy",
    ]
    if not all(path.exists() for path in required):
        return None

    nodes = pd.read_csv(processed_dir / "concept_nodes.csv")
    encounters = pd.read_csv(processed_dir / "encounter_hyperedges.csv")
    encounters["start_date"] = pd.to_datetime(encounters["start_date"], utc=True)
    encounters = encounters.sort_values(["patient_id", "start_date"]).reset_index(drop=True)
    edges = pd.read_csv(processed_dir / "node_encounter_edges.csv")

    node_ids = nodes["node_id"].tolist()
    encounter_ids = encounters["encounter_id"].tolist()
    patient_ids = encounters["patient_id"].tolist()
    pairs = list(zip(edges["node_id"].tolist(), edges["encounter_id"].tolist(), strict=False))

    x_v = torch.from_numpy(np.load(processed_dir / "node_embeddings.npy")).float()
    n_e = torch.from_numpy(np.load(processed_dir / "note_embeddings.npy")).float()
    if x_v.size(0) != len(node_ids) or n_e.size(0) != len(encounter_ids):
        raise ValueError("Embedding rows do not match hypergraph tables.")
    concept_semantics = x_v[:, -768:]

    inc_node, inc_edge, num_real, _ = build_augmented_incidence(node_ids, encounter_ids, pairs)
    labels, pair_mask = build_next_visit_labels(encounters, edges, node_ids)
    split_ids = split_patients(patient_ids, ratios, seed)
    return ProcessedBundle(
        node_ids=node_ids,
        encounter_ids=encounter_ids,
        patient_ids=patient_ids,
        node_states=x_v,
        concept_semantics=concept_semantics,
        note_semantics=n_e,
        incidence_node=inc_node,
        incidence_edge=inc_edge,
        num_real=num_real,
        num_self_loops=len(node_ids),
        labels=torch.from_numpy(labels),
        pair_mask=torch.from_numpy(pair_mask),
        split_ids=split_ids,
        source="processed",
    )


def make_synthetic_bundle(seed: int = 42) -> ProcessedBundle:
    """Small leak-checked graph for forward-pass validation (not full Coherent)."""
    rng = np.random.RandomState(seed)
    node_ids = [f"http://snomed.info/sct|{code}" for code, _ in FROZEN_TOP25[:8]]
    # Two patients, chronological visits; last visit of each is a terminal visit.
    encounter_ids = ["e0", "e1", "e2", "e3", "e4"]
    patient_ids = ["p0", "p0", "p0", "p1", "p1"]
    pairs = [
        (node_ids[0], "e0"),
        (node_ids[1], "e0"),
        (node_ids[0], "e1"),
        (node_ids[2], "e1"),
        (node_ids[3], "e2"),
        (node_ids[4], "e3"),
        (node_ids[5], "e4"),
    ]
    encounters = pd.DataFrame(
        {
            "encounter_id": encounter_ids,
            "patient_id": patient_ids,
            "start_date": pd.to_datetime(
                ["2020-01-01", "2020-02-01", "2020-03-01", "2020-01-01", "2020-06-01"], utc=True
            ),
        }
    )
    edges = pd.DataFrame(pairs, columns=["node_id", "encounter_id"])
    x_v = torch.from_numpy(rng.randn(len(node_ids), 832).astype(np.float32))
    n_e = torch.from_numpy(rng.randn(len(encounter_ids), 768).astype(np.float32))
    c_v = x_v[:, -768:]
    inc_node, inc_edge, num_real, _ = build_augmented_incidence(node_ids, encounter_ids, pairs)
    labels, pair_mask = build_next_visit_labels(encounters, edges, node_ids)
    split_ids = split_patients(patient_ids, (0.5, 0.5, 0.0), seed)
    return ProcessedBundle(
        node_ids=node_ids,
        encounter_ids=encounter_ids,
        patient_ids=patient_ids,
        node_states=x_v,
        concept_semantics=c_v,
        note_semantics=n_e,
        incidence_node=inc_node,
        incidence_edge=inc_edge,
        num_real=num_real,
        num_self_loops=len(node_ids),
        labels=torch.from_numpy(labels),
        pair_mask=torch.from_numpy(pair_mask),
        split_ids=split_ids,
        source="synthetic_forward_test",
    )
