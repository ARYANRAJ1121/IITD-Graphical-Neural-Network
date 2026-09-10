from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config
from src.data.processed_bundle import example_index, load_processed_bundle
from src.evaluation.ranking_metrics import class_row, macro_mean
from src.evaluation.threshold_analysis import CHECKPOINT, _load_model
from src.evaluation.thresholds import sigmoid
from src.graph.splits import build_eval_graph, patient_encounter_index
from src.training.train_vanilla_bce import _device, _forward, _masked_numpy, _split_logits, _to_device

REPORT = Path("experiments/vanilla_bce_ranking_diagnostic.md")


def _fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def _lift_fmt(value) -> str:
    if value is None:
        return "NA"
    return f"{value:.2f}×"


def write_report(path: Path, rows: list[dict], micro: dict, n_train: int, n_val: int, n_test: int, epoch) -> None:
    by_auprc = sorted(rows, key=lambda r: (-1.0 if r["auprc"] is None else -r["auprc"], r["rank"]))
    by_auroc = sorted(rows, key=lambda r: (-1.0 if r["auroc"] is None else -r["auroc"], r["rank"]))
    dialysis = rows[0]
    others = rows[1:]
    others_lift = [r["auprc_lift"] for r in others if r["auprc_lift"] is not None]
    others_auroc = [r["auroc"] for r in others if r["auroc"] is not None]
    n_lift_gt2 = sum(1 for r in rows if r["auprc_lift"] is not None and r["auprc_lift"] >= 2.0)
    n_auroc_gt06 = sum(1 for r in rows if r["auroc"] is not None and r["auroc"] >= 0.60)

    lines = [
        "# Vanilla BCE per-class ranking diagnostic",
        "",
        "Frozen checkpoint. No retraining, no architecture change, no class weights, no focal loss, no oversampling.",
        "No decision threshold is applied. Metrics are ranking-only.",
        "Aggregate micro-F1 is not used as a success criterion.",
        "",
        f"- checkpoint: `{CHECKPOINT}`",
        f"- checkpoint epoch: `{epoch}`",
        f"- examples: train `{n_train}`, val `{n_val}`, test `{n_test}`",
        "",
        "## Scientific question",
        "",
        "Does the frozen MINGLE model contain meaningful ranking signal across multiple target classes, or is almost all useful signal concentrated in renal dialysis?",
        "",
        "## Per-class results (test ranking)",
        "",
        "| # | Code | Name | Prev tr/va/te | Pos tr/va/te | AUROC | AUPRC | AUPRC/prev | mean p+ | mean p− | P@1 | R@1 | P@3 | R@3 | P@5 | R@5 |",
        "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        prev = f"{r['train_prev']:.4f}/{r['val_prev']:.4f}/{r['test_prev']:.4f}"
        pos = f"{r['train_pos']}/{r['val_pos']}/{r['test_pos']}"
        lines.append(
            f"| {r['rank']} | `{r['code']}` | {r['name']} | {prev} | {pos} | "
            f"{_fmt(r['auroc'])} | {_fmt(r['auprc'])} | {_lift_fmt(r['auprc_lift'])} | "
            f"{_fmt(r['mean_p_pos'])} | {_fmt(r['mean_p_neg'])} | "
            f"{_fmt(r['p_at_1'])} | {_fmt(r['r_at_1'])} | {_fmt(r['p_at_3'])} | {_fmt(r['r_at_3'])} | "
            f"{_fmt(r['p_at_5'])} | {_fmt(r['r_at_5'])} |"
        )

    lines.extend(
        [
            "",
            "## Averages",
            "",
            "| Metric | Macro (25 classes) | Micro (pooled test cells) |",
            "| --- | ---: | ---: |",
            f"| AUROC | {_fmt(macro_mean(rows, 'auroc'))} | {_fmt(micro['auroc'])} |",
            f"| AUPRC | {_fmt(macro_mean(rows, 'auprc'))} | {_fmt(micro['auprc'])} |",
            f"| AUPRC/prevalence | {_lift_fmt(macro_mean(rows, 'auprc_lift'))} | NA |",
            f"| mean p+ | {_fmt(macro_mean(rows, 'mean_p_pos'))} | NA |",
            f"| mean p− | {_fmt(macro_mean(rows, 'mean_p_neg'))} | NA |",
            f"| P@1 | {_fmt(macro_mean(rows, 'p_at_1'))} | NA |",
            f"| R@1 | {_fmt(macro_mean(rows, 'r_at_1'))} | NA |",
            f"| P@3 | {_fmt(macro_mean(rows, 'p_at_3'))} | NA |",
            f"| R@3 | {_fmt(macro_mean(rows, 'r_at_3'))} | NA |",
            f"| P@5 | {_fmt(macro_mean(rows, 'p_at_5'))} | NA |",
            f"| R@5 | {_fmt(macro_mean(rows, 'r_at_5'))} | NA |",
            "",
            "Micro P@k / R@k are omitted: those metrics are defined on a per-class ranked list.",
            "",
            "## Rank by test AUPRC",
            "",
            "| AUPRC rank | # | Name | AUPRC | AUPRC/prev | AUROC | test prev |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for i, r in enumerate(by_auprc, start=1):
        lines.append(
            f"| {i} | {r['rank']} | {r['name']} | {_fmt(r['auprc'])} | {_lift_fmt(r['auprc_lift'])} | {_fmt(r['auroc'])} | {r['test_prev']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Rank by test AUROC",
            "",
            "| AUROC rank | # | Name | AUROC | AUPRC | AUPRC/prev | test prev |",
            "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for i, r in enumerate(by_auroc, start=1):
        lines.append(
            f"| {i} | {r['rank']} | {r['name']} | {_fmt(r['auroc'])} | {_fmt(r['auprc'])} | {_lift_fmt(r['auprc_lift'])} | {r['test_prev']:.4f} |"
        )

    other_mean_lift = float(np.mean(others_lift)) if others_lift else None
    other_mean_auroc = float(np.mean(others_auroc)) if others_auroc else None
    lines.extend(
        [
            "",
            "## Comparison to prevalence baseline",
            "",
            "Random ranking AUPRC ≈ class prevalence. Lift = AUPRC / prevalence. Lift ≈ 1 means no ranking value.",
            "",
            f"- renal dialysis AUPRC `{_fmt(dialysis['auprc'])}` vs prevalence `{dialysis['test_prev']:.4f}` → lift `{_lift_fmt(dialysis['auprc_lift'])}`",
            f"- mean AUPRC lift over the other 24 classes: `{_lift_fmt(other_mean_lift)}`",
            f"- mean AUROC over the other 24 classes: `{_fmt(other_mean_auroc)}`",
            f"- classes with AUPRC lift ≥ 2: `{n_lift_gt2}/25`",
            f"- classes with AUROC ≥ 0.60: `{n_auroc_gt06}/25`",
            "",
            "## Answer",
            "",
        ]
    )
    lines.append(
        "**Almost all usable ranking mass is in renal dialysis**, with a short tail of weaker but real signal—"
        "not a flat 24-class null. Dialysis dominates absolute AUPRC; a few procedures have AUROC well above 0.5 "
        f"({n_lift_gt2}/25 with AUPRC lift ≥ 2; {n_auroc_gt06}/25 with AUROC ≥ 0.60), while most classes stay near the prevalence baseline. "
        "This is not a 25-class ranker and is not model success."
    )
    lines.append("")
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

    pair_mask = bundle.pair_mask.numpy()
    split_ex = example_index(bundle.patient_ids, pair_mask, bundle.split_ids)
    y_all = bundle.labels.numpy()
    y_train = y_all[split_ex["train"]]
    y_val = y_all[split_ex["val"]]
    y_test_labels = y_all[split_ex["test"]]

    device = _device()
    model, ckpt = _load_model(device, config)
    train_idx = patient_encounter_index(bundle.patient_ids, bundle.split_ids["train"])
    test_idx = patient_encounter_index(bundle.patient_ids, bundle.split_ids["test"])
    test_graph = _to_device(build_eval_graph(bundle, train_idx, test_idx, "test"), device)
    bundle.node_states = bundle.node_states.to(device)
    bundle.concept_semantics = bundle.concept_semantics.to(device)

    with torch.no_grad():
        te_logits = _forward(model, bundle, test_graph)
        te_logits, te_labels, te_mask = _split_logits(te_logits, test_graph)
    y_te, z_te = _masked_numpy(te_logits, te_labels, te_mask)
    if y_te.shape != y_test_labels.shape or not np.allclose(y_te, y_test_labels):
        raise RuntimeError("Test label alignment mismatch between split index and eval graph.")
    p_te = sigmoid(z_te)

    rows = [class_row(i, y_train, y_val, y_te, p_te) for i in range(25)]
    y_flat, p_flat = y_te.reshape(-1), p_te.reshape(-1)
    micro = {
        "auroc": float(roc_auc_score(y_flat, p_flat)),
        "auprc": float(average_precision_score(y_flat, p_flat)),
    }
    write_report(
        REPORT,
        rows,
        micro,
        n_train=int(y_train.shape[0]),
        n_val=int(y_val.shape[0]),
        n_test=int(y_te.shape[0]),
        epoch=ckpt.get("epoch"),
    )
    print(f"wrote {REPORT} dialysis_lift={rows[0]['auprc_lift']}", flush=True)


if __name__ == "__main__":
    main()
