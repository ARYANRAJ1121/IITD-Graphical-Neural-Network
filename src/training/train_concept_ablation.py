from __future__ import annotations

from pathlib import Path

import torch

from src.data.processed_bundle import ProcessedBundle
from src.training.train_vanilla_bce import train_vanilla_bce


def zero_concept_semantics(bundle: ProcessedBundle) -> None:
    """Drop C_v in both places it enters the model. Disk files are not written."""
    if bundle.concept_semantics.ndim != 2 or bundle.concept_semantics.size(1) != 768:
        raise RuntimeError(
            f"Expected concept semantics (V, 768); got {tuple(bundle.concept_semantics.shape)}"
        )
    if bundle.node_states.size(1) != 832:
        raise RuntimeError(
            f"Expected X_v dim 832 ([s_v; C_v]); got {tuple(bundle.node_states.shape)}"
        )
    bundle.concept_semantics = torch.zeros_like(bundle.concept_semantics)
    bundle.node_states = bundle.node_states.clone()
    bundle.node_states[:, -768:] = 0
    if not torch.equal(bundle.node_states[:, -768:], bundle.concept_semantics):
        raise RuntimeError("C_v zeroing failed: node_states tail != concept_semantics.")


def write_concept_ablation_report(path: Path, payload: dict) -> None:
    t = payload["test_metrics"]
    acc = t.get("accuracy")
    acc_s = f"{acc:.6f}" if acc is not None else "NA"
    lines = [
        "# Paper ablation: MINGLE w/o Medical Concept Semantics",
        "",
        "Parent = **vanilla BCE** (paper Eq. 4), same as full MINGLE. "
        "`C_v` zeroed in `X_v` and `N_aug`. DeepWalk and notes kept. 20 epochs.",
        "",
        f"- seed: `{payload['seed']}`",
        f"- best epoch: `{payload['best_epoch']}`",
        f"- device: `{payload['device']}`",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- ablation: `{payload.get('ablation')}`",
        "",
        "## Test",
        "",
        f"- ACC (Hamming @0.5): `{acc_s}`",
        f"- macro-AUROC: `{t['macro_auroc']}`",
        f"- macro-AUPRC: `{t['macro_auprc']}`",
        f"- micro-F1 @0.5: `{t['micro_f1']:.4f}`",
        f"- unweighted BCE: `{t['bce']:.6f}`",
        "",
        "See `experiments/paper_table1_coherent.md` for the three-row Table 1 analogue.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def train_concept_ablation(
    config: dict,
    bundle: ProcessedBundle,
    report_path: Path,
) -> dict:
    zero_concept_semantics(bundle)
    print(
        "Paper ablation w/o C_v: concept semantics zeroed; vanilla BCE. "
        f"{tuple(bundle.concept_semantics.shape)}. s_v and N_e unchanged.",
        flush=True,
    )
    payload = train_vanilla_bce(
        config,
        bundle,
        report_path,
        checkpoint_name="concept_ablation_best.pt",
        history_name="concept_ablation_history.json",
    )
    payload["ablation"] = "zero C_v in X_v and N_aug; vanilla BCE; DeepWalk and N_e unchanged"
    write_concept_ablation_report(report_path, payload)
    return payload
