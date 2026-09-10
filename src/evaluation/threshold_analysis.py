from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data.processed_bundle import load_processed_bundle
from src.evaluation.metrics import compute_metrics
from src.evaluation.thresholds import score_threshold, select_threshold, sigmoid
from src.graph.splits import build_eval_graph, patient_encounter_index
from src.models.mingle import MingleModel
from src.training.train_vanilla_bce import (
    _device,
    _forward,
    _masked_numpy,
    _split_logits,
    _to_device,
)

THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
CHECKPOINT = Path("data/processed/checkpoints/vanilla_bce_best.pt")
REPORT = Path("experiments/vanilla_bce_threshold_analysis.md")


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _load_model(device: torch.device, config: dict) -> MingleModel:
    frozen = config["model"]
    model = MingleModel(
        node_input_dim=int(frozen["node_input_dim"]),
        semantic_dim=int(frozen["semantic_dim"]),
        hidden_dim=48,
        num_heads=4,
        num_layers=2,
        num_classes=25,
        expansion=int(frozen["ffn_expansion"]),
        pairnorm_eps=float(frozen["pairnorm_eps"]),
    ).to(device)
    try:
        ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def _prob_summary(probs: np.ndarray) -> dict:
    flat = probs.reshape(-1)
    return {
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "p50": float(np.quantile(flat, 0.50)),
        "p90": float(np.quantile(flat, 0.90)),
        "p99": float(np.quantile(flat, 0.99)),
        "frac_ge_0.5": float((flat >= 0.5).mean()),
    }


def write_report(
    path: Path,
    *,
    ckpt: dict,
    val_rows: list[dict],
    test_rows: list[dict],
    selected: dict,
    test_at_selected: dict,
    val_rank: dict,
    test_rank: dict,
    val_prob: dict,
    test_prob: dict,
) -> None:
    test_by_t = {row["threshold"]: row for row in test_rows}
    lines = [
        "# Vanilla BCE threshold sensitivity",
        "",
        "Checkpoint frozen. No retraining. Architecture, loss, and optimizer unchanged.",
        "Threshold chosen on **validation only**. Test is scored once at that threshold.",
        "A higher F1 at a lower threshold is not treated as model success by itself.",
        "",
        f"- checkpoint: `{CHECKPOINT}`",
        f"- checkpoint epoch: `{ckpt.get('epoch')}`",
        f"- checkpoint val BCE: `{ckpt.get('val_bce')}`",
        "- selection rule: maximize validation **micro-F1**; ties broken by validation macro-F1, then by the larger threshold",
        f"- selected threshold: **{selected['threshold']}**",
        "",
        "## Threshold-independent ranking (unchanged)",
        "",
        "| Split | macro-AUROC | micro-AUPRC | macro-AUPRC |",
        "| --- | ---: | ---: | ---: |",
        f"| validation | {val_rank['macro_auroc']} | {val_rank['micro_auprc']} | {val_rank['macro_auprc']} |",
        f"| test | {test_rank['macro_auroc']} | {test_rank['micro_auprc']} | {test_rank['macro_auprc']} |",
        "",
        "## Predicted probability calibration (sigmoid logits)",
        "",
        "| Split | mean | std | min | max | p50 | p90 | p99 | share ≥ 0.5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| validation | {val_prob['mean']:.6f} | {val_prob['std']:.6f} | {val_prob['min']:.6f} | {val_prob['max']:.6f} | {val_prob['p50']:.6f} | {val_prob['p90']:.6f} | {val_prob['p99']:.6f} | {100*val_prob['frac_ge_0.5']:.4f}% |",
        f"| test | {test_prob['mean']:.6f} | {test_prob['std']:.6f} | {test_prob['min']:.6f} | {test_prob['max']:.6f} | {test_prob['p50']:.6f} | {test_prob['p90']:.6f} | {test_prob['p99']:.6f} | {100*test_prob['frac_ge_0.5']:.4f}% |",
        "",
        "## Validation sweep (used for selection)",
        "",
        "| τ | micro-F1 | macro-F1 | micro-P | micro-R | macro-P | macro-R | # pos | % pos |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in val_rows:
        mark = " ← selected" if row["threshold"] == selected["threshold"] else ""
        lines.append(
            f"| {row['threshold']:.3f} | {_fmt(row['micro_f1'])} | {_fmt(row['macro_f1'])} | "
            f"{_fmt(row['micro_precision'])} | {_fmt(row['micro_recall'])} | "
            f"{_fmt(row['macro_precision'])} | {_fmt(row['macro_recall'])} | "
            f"{row['n_positive_predictions']} | {row['pct_positive_predictions']:.4f}%{mark} |"
        )

    lines.extend(
        [
            "",
            "## Test sweep (reference only; not used for selection)",
            "",
            "| τ | micro-F1 | macro-F1 | micro-P | micro-R | macro-P | macro-R | # pos | % pos |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in test_rows:
        lines.append(
            f"| {row['threshold']:.3f} | {_fmt(row['micro_f1'])} | {_fmt(row['macro_f1'])} | "
            f"{_fmt(row['micro_precision'])} | {_fmt(row['micro_recall'])} | "
            f"{_fmt(row['macro_precision'])} | {_fmt(row['macro_recall'])} | "
            f"{row['n_positive_predictions']} | {row['pct_positive_predictions']:.4f}% |"
        )

    tau = selected["threshold"]
    locked = test_by_t[tau]
    lines.extend(
        [
            "",
            f"## Locked test evaluation at validation-selected τ = {tau}",
            "",
            f"- micro-F1: `{locked['micro_f1']:.4f}`",
            f"- macro-F1: `{locked['macro_f1']:.4f}`",
            f"- micro-precision: `{locked['micro_precision']:.4f}`",
            f"- micro-recall: `{locked['micro_recall']:.4f}`",
            f"- macro-precision: `{locked['macro_precision']:.4f}`",
            f"- macro-recall: `{locked['macro_recall']:.4f}`",
            f"- positive predictions: `{locked['n_positive_predictions']}` ({locked['pct_positive_predictions']:.4f}% of label cells)",
            "",
            "| Rank | Code | Name | Precision | Recall | F1 | Support |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in locked["per_class"]:
        lines.append(
            f"| {row['rank']} | `{row['code']}` | {row['name']} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | {row['support']} |"
        )

    t05 = next(r for r in val_rows if r["threshold"] == 0.5)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"At τ=0.50, validation micro-F1 is `{t05['micro_f1']:.4f}` with `{t05['n_positive_predictions']}` positive predictions "
            f"({t05['pct_positive_predictions']:.4f}% of cells). Mean predicted probability is `{val_prob['mean']:.6f}` "
            f"(p99 `{val_prob['p99']:.6f}`), so a 0.5 cut sits far above the model's score mass.",
            "",
            "If F1 rises at lower thresholds while AUROC/AUPRC stay fixed, the 0.5-threshold zero-F1 is consistent with "
            "**miscalibration / imbalance operating-point mismatch**, not with a complete absence of ranking signal.",
            "",
            "That does **not** make the run successful: a lower threshold also increases false positives, and F1 can be "
            "inflated by over-predicting. Compare AUPRC and per-class F1 before claiming utility.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not CHECKPOINT.exists():
        raise SystemExit(f"Missing checkpoint: {CHECKPOINT}")
    config = load_config(Path("configs/base.yaml"))
    bundle = load_processed_bundle(
        Path(config["data"]["processed_dir"]),
        int(config["training"]["seed"]),
        tuple(config["training"]["patient_split"]),
    )
    if bundle is None:
        raise SystemExit("Processed tensors are missing.")

    device = _device()
    model, ckpt = _load_model(device, config)
    train_idx = patient_encounter_index(bundle.patient_ids, bundle.split_ids["train"])
    val_idx = patient_encounter_index(bundle.patient_ids, bundle.split_ids["val"])
    test_idx = patient_encounter_index(bundle.patient_ids, bundle.split_ids["test"])
    val_graph = _to_device(build_eval_graph(bundle, train_idx, val_idx, "val"), device)
    test_graph = _to_device(build_eval_graph(bundle, train_idx, test_idx, "test"), device)
    bundle.node_states = bundle.node_states.to(device)
    bundle.concept_semantics = bundle.concept_semantics.to(device)

    with torch.no_grad():
        va_logits = _forward(model, bundle, val_graph)
        te_logits = _forward(model, bundle, test_graph)
        va_logits, va_labels, va_mask = _split_logits(va_logits, val_graph)
        te_logits, te_labels, te_mask = _split_logits(te_logits, test_graph)

    y_va, z_va = _masked_numpy(va_logits, va_labels, va_mask)
    y_te, z_te = _masked_numpy(te_logits, te_labels, te_mask)
    p_va = sigmoid(z_va)
    p_te = sigmoid(z_te)

    val_rank = compute_metrics(y_va, z_va, p_va)
    test_rank = compute_metrics(y_te, z_te, p_te)
    val_rows = [score_threshold(y_va, p_va, t) for t in THRESHOLDS]
    test_rows = [score_threshold(y_te, p_te, t) for t in THRESHOLDS]
    selected = select_threshold(val_rows)
    test_at_selected = next(r for r in test_rows if r["threshold"] == selected["threshold"])

    write_report(
        REPORT,
        ckpt=ckpt,
        val_rows=val_rows,
        test_rows=test_rows,
        selected=selected,
        test_at_selected=test_at_selected,
        val_rank=val_rank,
        test_rank=test_rank,
        val_prob=_prob_summary(p_va),
        test_prob=_prob_summary(p_te),
    )
    print(
        f"selected_threshold={selected['threshold']} "
        f"val_micro_f1={selected['micro_f1']:.4f} "
        f"test_micro_f1={test_at_selected['micro_f1']:.4f} "
        f"report={REPORT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
