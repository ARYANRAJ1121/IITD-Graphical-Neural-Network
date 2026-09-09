from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.data.stage3_bundle import Stage3Bundle
from src.training.train_stage4a import train_stage4a
from src.training.train_stage4b import STAGE4A50_TEST


def write_stage5a_report(path: Path, payload: dict) -> None:
    s4 = STAGE4A50_TEST
    t = payload["test_metrics"]
    v = payload["val_metrics"]
    tr = payload["train_metrics"]
    weights = payload["pos_weights"]
    rest = [row["auprc"] for row in t["per_class"][1:] if row["auprc"] is not None]
    d_macro_auprc = t["macro_auprc"] - s4["macro_auprc"]
    d_micro_auprc = t["micro_auprc"] - s4["micro_auprc"]
    d_macro_auroc = t["macro_auroc"] - s4["macro_auroc"]
    d_micro_f1 = t["micro_f1"] - s4["micro_f1"]
    d_macro_f1 = t["macro_f1"] - s4["macro_f1"]
    n_gt_4a = sum(
        1
        for i, row in enumerate(t["per_class"])
        if row["auprc"] is not None and row["auprc"] > s4["per_class_auprc"][i]
    )
    n_lt_4a = sum(
        1
        for i, row in enumerate(t["per_class"])
        if row["auprc"] is not None and row["auprc"] < s4["per_class_auprc"][i]
    )

    notes_help = (
        t["macro_auprc"] < s4["macro_auprc"] and t["micro_auprc"] < s4["micro_auprc"]
    )
    notes_hurt = (
        t["macro_auprc"] > s4["macro_auprc"] and t["micro_auprc"] > s4["micro_auprc"]
    )
    if notes_help:
        conclusion = (
            "Removing clinical-note semantics **lowers** both macro-AUPRC and micro-AUPRC "
            "vs Stage 4A, so note embeddings contributed useful signal on this Coherent run."
        )
    elif notes_hurt:
        conclusion = (
            "Removing clinical-note semantics **raises** both macro-AUPRC and micro-AUPRC "
            "vs Stage 4A. Do not treat notes as helpful on this evidence."
        )
    else:
        conclusion = (
            "Removing notes does **not** move macro-AUPRC and micro-AUPRC in the same "
            "direction vs Stage 4A. Do not claim that notes help or hurt overall."
        )

    lines = [
        "# Stage 5A: No Clinical Note Semantics",
        "",
        "## Objective",
        "",
        "Ablate **clinical note embeddings** (`N_e`) only, with Stage 4A weighted BCE and the "
        "frozen MINGLE architecture. Compare full model "
        "(DeepWalk + concept semantics + note semantics) vs "
        "(DeepWalk + concept semantics + no note information).",
        "",
        "## How note information was removed",
        "",
        "Notes enter only as `graph.notes` → `MingleModel.forward(note_semantics)` → "
        "`N_aug = cat([N_e, C_v])` → `H_e = MLP_1(N_aug)`.",
        "Stage 5A replaces in-memory `bundle.note_semantics` with **zeros of shape `(E, 768)`** "
        "before graphs are built. `note_embeddings.npy` on disk is not modified. "
        "Not random. Concept rows of `N_aug` remain `C_v`. DeepWalk `X_v` is unchanged. "
        "Because MLP_1 is `Linear`, real-visit `H_e` becomes the shared bias vector; visits "
        "still differ through PMA over incident concepts (`A_e`).",
        "",
        "## Exact experimental setup",
        "",
        f"- seed: `{payload['seed']}`",
        f"- split: `{payload['split']}`",
        f"- epochs (max): `{payload['epochs']}`",
        f"- optimizer: Adam, lr `{payload['lr']}` (no weight decay)",
        f"- loss: class-weighted BCE, train-only `n_neg/n_pos`",
        f"- d=48, h=4, L=2, 25 classes, PMA + PairNorm + MLP_1/2 + self-loops + JK-CONCAT",
        f"- parameter count: `{payload['param_count']}`",
        f"- device: `{payload['device']}`",
        f"- hardware: `{payload['hardware']}`",
        f"- runtime_sec: `{payload['runtime_sec']:.1f}`",
        f"- best epoch: `{payload['best_epoch']}` (lowest unweighted val BCE)",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- early stopping: `{payload.get('early_stopping', 'off')}`",
        f"- note ablation: `{payload.get('ablation', 'zero N_e')}`",
        "",
        "Unchanged vs Stage 4A: architecture, graph topology, DeepWalk, concept semantics, "
        "labels, split, seed, Adam, lr, weighted BCE, evaluation, checkpoint rule.",
        "",
        "## Train-only class weights (`n_neg / n_pos`)",
        "",
        "| Rank | pos_weight | train pos |",
        "| ---: | ---: | ---: |",
    ]
    for i, w in enumerate(weights):
        lines.append(f"| {i + 1} | {w:.4f} | {payload['train_pos'][i]} |")
    lines.extend(
        [
            "",
            "## Training / convergence",
            "",
            "| Epoch | Train weighted BCE | Train unweighted BCE | Val weighted BCE | Val unweighted BCE |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["history"]:
        vw = row.get("val_weighted_bce", float("nan"))
        lines.append(
            f"| {row['epoch']} | {row['train_weighted_bce']:.6f} | {row['train_bce']:.6f} | "
            f"{vw:.6f} | {row['val_bce']:.6f} |"
        )

    def split_block(title: str, m: dict) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                f"- n examples: `{m['n_examples']}`",
                f"- unweighted BCE: `{m['bce']:.6f}`",
                f"- micro-AUPRC: `{m['micro_auprc']}`",
                f"- macro-AUPRC: `{m['macro_auprc']}`",
                f"- macro-AUROC: `{m['macro_auroc']}`",
                f"- micro-F1 @0.5: `{m['micro_f1']:.4f}`",
                f"- macro-F1 @0.5: `{m['macro_f1']:.4f}`",
                f"- positive predictions @0.5: `{m.get('n_positive_predictions', 'NA')}` "
                f"({m.get('pct_positive_predictions', 0):.4f}% of cells)",
                "",
                "| Rank | Code | Name | AUROC | AUPRC | F1@0.5 | P | R | Support |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in m["per_class"]:
            lines.append(
                f"| {row['rank']} | `{row['code']}` | {row['name']} | {row['auroc']} | {row['auprc']} | "
                f"{row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['support']} |"
            )

    split_block("Train (best checkpoint)", tr)
    split_block("Validation (best checkpoint)", v)
    split_block("Test (best checkpoint)", t)
    lines.extend(
        [
            "",
            "## Stage 4A vs Stage 5A (test)",
            "",
            "| Metric | Stage 4A Full + Weighted BCE | Stage 5A No notes + Weighted BCE | Δ (5A − 4A) |",
            "| --- | ---: | ---: | ---: |",
            f"| unweighted BCE | {s4['bce']:.6f} | {t['bce']:.6f} | {t['bce']-s4['bce']:.6f} |",
            f"| micro-AUPRC | {s4['micro_auprc']:.6f} | {t['micro_auprc']:.6f} | {d_micro_auprc:.6f} |",
            f"| macro-AUPRC | {s4['macro_auprc']:.6f} | {t['macro_auprc']:.6f} | {d_macro_auprc:.6f} |",
            f"| macro-AUROC | {s4['macro_auroc']:.6f} | {t['macro_auroc']:.6f} | {d_macro_auroc:.6f} |",
            f"| micro-F1 @0.5 | {s4['micro_f1']:.4f} | {t['micro_f1']:.4f} | {d_micro_f1:.4f} |",
            f"| macro-F1 @0.5 | {s4['macro_f1']:.4f} | {t['macro_f1']:.4f} | {d_macro_f1:.4f} |",
            f"| # pos preds @0.5 | {s4['n_positive_predictions']} | {t.get('n_positive_predictions', 0)} | |",
            "",
            f"- change in macro-AUPRC: `{d_macro_auprc:.6f}`",
            f"- change in micro-AUPRC: `{d_micro_auprc:.6f}`",
            f"- change in macro-AUROC: `{d_macro_auroc:.6f}`",
            f"- change in micro-F1: `{d_micro_f1:.4f}`",
            f"- change in macro-F1: `{d_macro_f1:.4f}`",
            f"- classes with AUPRC > Stage 4A: `{n_gt_4a}/25`",
            f"- classes with AUPRC < Stage 4A: `{n_lt_4a}/25`",
            "",
            "## Per-class AUPRC vs Stage 4A",
            "",
            "| Rank | Name | 4A AUPRC | 5A AUPRC | Δ |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for i, row in enumerate(t["per_class"]):
        a4 = s4["per_class_auprc"][i]
        a5 = row["auprc"] or 0.0
        lines.append(f"| {row['rank']} | {row['name']} | {a4:.4f} | {a5:.4f} | {a5-a4:.4f} |")
    mean_rest = float(np.mean(rest)) if rest else float("nan")
    lines.extend(
        [
            "",
            f"- dialysis test AUPRC: `{t['per_class'][0]['auprc']}`",
            f"- mean AUPRC of other 24 classes: `{mean_rest}`",
            "",
            "## Interpretation",
            "",
            payload["dominance_note"],
            "",
            conclusion,
            "",
            "## Limitations",
            "",
            "- Single seed. Zeros are a shared MLP_1 bias for all visits, not a removed MLP_1 channel.",
            "- Stage 4A numbers are the stored 50-epoch converged test metrics; 4A was not retrained.",
            "- Coherent notes are synthetic; empty-note visits already existed in Stage 2.",
            "",
            "## Numerical issues",
            "",
            payload.get("numerical_notes", "None recorded."),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def train_stage5a(
    config: dict,
    bundle: Stage3Bundle,
    report_path: Path,
) -> dict:
    if bundle.note_semantics.ndim != 2 or bundle.note_semantics.size(1) != 768:
        raise RuntimeError(
            f"Expected note semantics (E, 768); got {tuple(bundle.note_semantics.shape)}"
        )
    if bundle.concept_semantics.size(1) != 768:
        raise RuntimeError("Concept semantics dim changed; refusing ablation.")

    # Neutral notes: zeros, same dtype/shape. Disk embeddings are not written.
    bundle.note_semantics = torch.zeros_like(bundle.note_semantics)
    print(
        "Stage 5A: N_e replaced with zeros "
        f"{tuple(bundle.note_semantics.shape)}. C_v / X_v / incidence unchanged.",
        flush=True,
    )

    payload = train_stage4a(
        config,
        bundle,
        report_path,
        epochs=50,
        checkpoint_name="stage5a_no_note_semantics_best.pt",
        history_name="stage5a_no_note_semantics_history.json",
        early_stop_patience=10,
        experiment_name="stage5a_no_note_semantics",
        compare_stage4a20=False,
    )
    payload["ablation"] = "zero N_e (in-memory only); C_v and DeepWalk unchanged"
    write_stage5a_report(report_path, payload)
    return payload
