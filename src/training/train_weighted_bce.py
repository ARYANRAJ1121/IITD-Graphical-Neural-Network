from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.data.processed_bundle import ProcessedBundle
from src.data.next_visit_labels import FROZEN_TOP25
from src.evaluation.metrics import compute_metrics
from src.graph.splits import build_eval_graph, build_split_graph, patient_encounter_index
from src.models.mingle import MingleModel
from src.training.train_vanilla_bce import (
    _check_finite,
    _cuda_mem,
    _device,
    _eval_forward,
    _forward,
    _masked_numpy,
    _split_logits,
    _to_device,
)
from src.training.validate_architecture import masked_bce_with_logits

VANILLA_BCE_TEST = {
    "bce": 0.062200,
    "micro_f1": 0.0,
    "macro_f1": 0.0,
    "macro_auroc": 0.6545617075251373,
    "micro_auprc": 0.15783638442963063,
    "macro_auprc": 0.06427900818886663,
    "n_positive_predictions": 0,
    "per_class_auprc": [
        0.8130, 0.1151, 0.3146, 0.0305, 0.1394, 0.0248, 0.0101, 0.0165, 0.0148, 0.0126,
        0.0089, 0.0067, 0.0082, 0.0049, 0.0068, 0.0051, 0.0138, 0.0039, 0.0110, 0.0061,
        0.0151, 0.0060, 0.0039, 0.0100, 0.0052,
    ],
    "per_class_auroc": [
        0.9566, 0.5762, 0.7756, 0.6318, 0.8624, 0.7045, 0.5575, 0.6979, 0.6894, 0.7223,
        0.5769, 0.5476, 0.6232, 0.6005, 0.5922, 0.4127, 0.8200, 0.5381, 0.8326, 0.5840,
        0.6153, 0.6203, 0.5716, 0.6616, 0.5932,
    ],
    "per_class_f1": [0.0] * 25,
}


WEIGHTED_BCE_20_TEST = {
    "bce": 0.415957,
    "micro_f1": 0.1029,
    "macro_f1": 0.1249,
    "macro_auroc": 0.880121,
    "micro_auprc": 0.150431,
    "macro_auprc": 0.236658,
    "mean_other24": 0.211150,
    "best_epoch": 20,
}


def _append_convergence_comparison(path: Path, payload: dict) -> None:
    t = payload["test_metrics"]
    rest = [row["auprc"] for row in t["per_class"][1:] if row["auprc"] is not None]
    mean_rest = float(np.mean(rest)) if rest else float("nan")
    s3 = VANILLA_BCE_TEST
    s4 = WEIGHTED_BCE_20_TEST
    n_gt_s3 = sum(
        1
        for i, row in enumerate(t["per_class"])
        if row["auprc"] is not None and row["auprc"] > s3["per_class_auprc"][i]
    )
    extra = [
        "",
        "## Comparison vs Vanilla BCE and Weighted BCE (20-epoch)",
        "",
        "| Metric | Vanilla BCE vanilla | Weighted BCE 20-epoch | Weighted BCE converged | vs Vanilla BCE | vs WBCE-20 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| unweighted test BCE | {s3['bce']:.6f} | {s4['bce']:.6f} | {t['bce']:.6f} | {t['bce']-s3['bce']:.6f} | {t['bce']-s4['bce']:.6f} |",
        f"| micro-AUPRC | {s3['micro_auprc']:.6f} | {s4['micro_auprc']:.6f} | {t['micro_auprc']:.6f} | {t['micro_auprc']-s3['micro_auprc']:.6f} | {t['micro_auprc']-s4['micro_auprc']:.6f} |",
        f"| macro-AUPRC | {s3['macro_auprc']:.6f} | {s4['macro_auprc']:.6f} | {t['macro_auprc']:.6f} | {t['macro_auprc']-s3['macro_auprc']:.6f} | {t['macro_auprc']-s4['macro_auprc']:.6f} |",
        f"| macro-AUROC | {s3['macro_auroc']:.6f} | {s4['macro_auroc']:.6f} | {t['macro_auroc']:.6f} | {t['macro_auroc']-s3['macro_auroc']:.6f} | {t['macro_auroc']-s4['macro_auroc']:.6f} |",
        f"| micro-F1 @0.5 | {s3['micro_f1']:.4f} | {s4['micro_f1']:.4f} | {t['micro_f1']:.4f} | {t['micro_f1']-s3['micro_f1']:.4f} | {t['micro_f1']-s4['micro_f1']:.4f} |",
        f"| macro-F1 @0.5 | {s3['macro_f1']:.4f} | {s4['macro_f1']:.4f} | {t['macro_f1']:.4f} | {t['macro_f1']-s3['macro_f1']:.4f} | {t['macro_f1']-s4['macro_f1']:.4f} |",
        f"| mean AUPRC other 24 | — | {s4['mean_other24']:.4f} | {mean_rest:.4f} | — | {mean_rest-s4['mean_other24']:.4f} |",
        "",
        f"- classes with test AUPRC > Vanilla BCE: `{n_gt_s3}/25`",
        f"- dialysis AUPRC: `{t['per_class'][0]['auprc']}`",
        "",
        "## Convergence questions",
        "",
        f"- A. macro-AUPRC still ≥ 0.20? **{'yes' if t['macro_auprc'] and t['macro_auprc'] >= 0.20 else 'no'}** (`{t['macro_auprc']}`)",
        f"- B. most classes still above Vanilla BCE AUPRC? **{'yes' if n_gt_s3 >= 13 else 'no'}** (`{n_gt_s3}/25`)",
        f"- C. distributed vs dialysis-only? mean other-24 AUPRC `{mean_rest:.4f}` vs dialysis `{t['per_class'][0]['auprc']}`",
        "",
    ]
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(extra))


def train_pos_weights(train_labels: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
    """pos_weight_c = n_neg_c / n_pos_c from training pairs only."""
    y = train_labels[pair_mask].float()
    n_pos = y.sum(dim=0)
    n_neg = y.size(0) - n_pos
    if torch.any(n_pos <= 0):
        raise RuntimeError("A training class has zero positives; cannot form pos_weight.")
    return n_neg / n_pos


def masked_weighted_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pair_mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    return criterion(logits[pair_mask], labels[pair_mask])


def write_weighted_bce_report(path: Path, payload: dict) -> None:
    s3 = VANILLA_BCE_TEST
    t = payload["test_metrics"]
    v = payload["val_metrics"]
    tr = payload["train_metrics"]
    weights = payload["pos_weights"]
    dialysis_auprc = t["per_class"][0]["auprc"]
    rest = [row["auprc"] for row in t["per_class"][1:] if row["auprc"] is not None]
    lines = [
        f"# Weighted BCE ({payload.get('experiment', 'weighted_bce')})",
        "",
        "Isolated loss-function experiment. Architecture, graph, embeddings, labels, split, and seed are frozen.",
        "Positive-class weights from **training pairs only**. No focal loss, oversampling, or threshold tuning during training.",
        "F1 uses the paper's 0.5 threshold. Checkpoint selected by **unweighted** validation BCE (same rule as Vanilla BCE).",
        "",
        "## Setup",
        "",
        f"- seed: `{payload['seed']}`",
        f"- split: `{payload['split']}`",
        f"- epochs: `{payload['epochs']}`",
        f"- optimizer: Adam, lr `{payload['lr']}`",
        f"- d=48, h=4, L=2, 25 classes, PMA + PairNorm + MLP_1/2 + self-loops + JK-CONCAT",
        f"- parameter count: `{payload['param_count']}`",
        f"- device: `{payload['device']}`",
        f"- hardware: `{payload['hardware']}`",
        f"- runtime_sec: `{payload['runtime_sec']:.1f}`",
        f"- best epoch: `{payload['best_epoch']}` (lowest unweighted val BCE)",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- early stopping: `{payload.get('early_stopping', 'off')}`",
        "",
        "## Train-only class weights (`n_neg / n_pos`)",
        "",
        "| Rank | Code | Name | pos_weight | train pos |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    for i, (code, name) in enumerate(FROZEN_TOP25):
        lines.append(
            f"| {i + 1} | `{code}` | {name} | {weights[i]:.4f} | {payload['train_pos'][i]} |"
        )
    lines.extend(
        [
            "",
            "## Epoch losses",
            "",
            "| Epoch | Train weighted BCE | Train unweighted BCE | Val weighted BCE | Val unweighted BCE |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in payload["history"]:
        vw = row.get("val_weighted_bce", float("nan"))
        lines.append(
            f"| {row['epoch']} | {row['train_weighted_bce']:.6f} | {row['train_bce']:.6f} | {vw:.6f} | {row['val_bce']:.6f} |"
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
                f"- positive predictions @0.5: `{m.get('n_positive_predictions', 'NA')}` ({m.get('pct_positive_predictions', 0):.4f}% of cells)",
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
            "## Comparison vs Vanilla BCE vanilla BCE (test)",
            "",
            "| Metric | Vanilla BCE vanilla | Weighted BCE weighted | Δ |",
            "| --- | ---: | ---: | ---: |",
            f"| BCE | {s3['bce']:.6f} | {t['bce']:.6f} | {t['bce'] - s3['bce']:.6f} |",
            f"| micro-AUPRC | {s3['micro_auprc']:.6f} | {t['micro_auprc']:.6f} | {t['micro_auprc'] - s3['micro_auprc']:.6f} |",
            f"| macro-AUPRC | {s3['macro_auprc']:.6f} | {t['macro_auprc']:.6f} | {t['macro_auprc'] - s3['macro_auprc']:.6f} |",
            f"| macro-AUROC | {s3['macro_auroc']:.6f} | {t['macro_auroc']:.6f} | {t['macro_auroc'] - s3['macro_auroc']:.6f} |",
            f"| micro-F1 @0.5 | {s3['micro_f1']:.4f} | {t['micro_f1']:.4f} | {t['micro_f1'] - s3['micro_f1']:.4f} |",
            f"| macro-F1 @0.5 | {s3['macro_f1']:.4f} | {t['macro_f1']:.4f} | {t['macro_f1'] - s3['macro_f1']:.4f} |",
            f"| # pos preds @0.5 | {s3['n_positive_predictions']} | {t.get('n_positive_predictions', 0)} | |",
            "",
            "| Rank | Name | S3 AUROC | S4 AUROC | S3 AUPRC | S4 AUPRC | S3 F1 | S4 F1 |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    n_improve_auprc = 0
    for i, row in enumerate(t["per_class"]):
        a3, r3, f3 = s3["per_class_auprc"][i], s3["per_class_auroc"][i], s3["per_class_f1"][i]
        a4, r4, f4 = row["auprc"] or 0.0, row["auroc"] or 0.0, row["f1"]
        if row["auprc"] is not None and row["auprc"] > a3:
            n_improve_auprc += 1
        lines.append(
            f"| {row['rank']} | {row['name']} | {r3:.4f} | {r4:.4f} | {a3:.4f} | {a4:.4f} | {f3:.4f} | {f4:.4f} |"
        )
    dial_share = (dialysis_auprc or 0) / t["macro_auprc"] if t["macro_auprc"] else 0
    lines.extend(
        [
            "",
            "## Is improvement distributed or dialysis-dominated?",
            "",
            f"- dialysis test AUPRC: `{dialysis_auprc}`",
            f"- mean AUPRC of other 24 classes: `{float(np.mean(rest)) if rest else 'NA'}`",
            f"- classes with AUPRC > Vanilla BCE: `{n_improve_auprc}/25`",
            f"- dialysis AUPRC / macro-AUPRC: `{dial_share:.2f}`",
            "",
            payload["dominance_note"],
            "",
            "## Numerical issues",
            "",
            payload.get("numerical_notes", "None recorded."),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def train_weighted_bce(
    config: dict,
    bundle: ProcessedBundle,
    report_path: Path,
    *,
    epochs: int | None = None,
    checkpoint_name: str = "weighted_bce_best.pt",
    history_name: str = "weighted_bce_history.json",
    early_stop_patience: int | None = None,
    experiment_name: str = "weighted_bce",
    compare_weighted_bce_20: bool = False,
) -> dict:
    if bundle.source != "processed":
        raise RuntimeError("Refusing Weighted BCE on synthetic data.")
    frozen = config["model"]
    if (
        frozen["hidden_dim"] != 48
        or frozen["num_heads"] != 4
        or frozen["num_layers"] != 2
        or frozen["num_classes"] != 25
    ):
        raise RuntimeError("Frozen architecture constants were altered.")

    device = _device()
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(
        f"Weighted BCE weighted BCE on {device}. Frozen architecture. experiment={experiment_name}",
        flush=True,
    )

    train_idx = patient_encounter_index(bundle.patient_ids, bundle.split_ids["train"])
    val_idx = patient_encounter_index(bundle.patient_ids, bundle.split_ids["val"])
    test_idx = patient_encounter_index(bundle.patient_ids, bundle.split_ids["test"])
    train_graph = _to_device(build_split_graph(bundle, train_idx, "train"), device)
    val_graph = build_eval_graph(bundle, train_idx, val_idx, "val")
    test_graph = build_eval_graph(bundle, train_idx, test_idx, "test")
    bundle.node_states = bundle.node_states.to(device)
    bundle.concept_semantics = bundle.concept_semantics.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _cuda_mem("after train graph + embeddings", device)

    pos_weight = train_pos_weights(train_graph.labels, train_graph.pair_mask).to(device)
    train_pos = train_graph.labels[train_graph.pair_mask].sum(dim=0).detach().cpu().tolist()
    print("train pos_weight=" + ",".join(f"{w:.3f}" for w in pos_weight.tolist()), flush=True)

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
    param_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["lr"]))
    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / checkpoint_name

    epochs = int(epochs if epochs is not None else config["training"]["epochs"])
    history = []
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    stopped_early = False
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = _forward(model, bundle, train_graph)
        train_w = masked_weighted_bce(train_logits, train_graph.labels, train_graph.pair_mask, pos_weight)
        _check_finite(train_w, epoch)
        train_w.backward()
        optimizer.step()
        train_w_item = float(train_w.item())
        with torch.no_grad():
            train_u = masked_bce_with_logits(train_logits, train_graph.labels, train_graph.pair_mask)
            train_u_item = float(train_u.item())
        del train_logits, train_w, train_u
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.eval()
        val_logits = _eval_forward(model, bundle, val_graph, device)
        val_logits, val_labels, val_mask = _split_logits(val_logits, val_graph)
        val_u = masked_bce_with_logits(val_logits, val_labels, val_mask)
        val_w = masked_weighted_bce(val_logits, val_labels, val_mask, pos_weight)
        _check_finite(val_u, epoch)
        history.append(
            {
                "epoch": epoch,
                "train_weighted_bce": train_w_item,
                "train_bce": train_u_item,
                "val_weighted_bce": float(val_w.item()),
                "val_bce": float(val_u.item()),
            }
        )
        print(
            f"epoch {epoch}/{epochs} train_w={train_w_item:.6f} train_u={train_u_item:.6f} "
            f"val_w={val_w.item():.6f} val_u={val_u.item():.6f}",
            flush=True,
        )
        if float(val_u.item()) < best_val:
            best_val = float(val_u.item())
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_bce": best_val,
                    "pos_weight": pos_weight.detach().cpu(),
                    "experiment": experiment_name,
                },
                best_path,
            )
        else:
            stale += 1
            if early_stop_patience is not None and stale >= early_stop_patience:
                stopped_early = True
                print(
                    f"Early stop at epoch {epoch}: unweighted val BCE did not improve for {early_stop_patience} epochs.",
                    flush=True,
                )
                break

    runtime = time.perf_counter() - started
    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        tr_logits = _forward(model, bundle, train_graph)
    va_logits = _eval_forward(model, bundle, val_graph, device)
    te_logits = _eval_forward(model, bundle, test_graph, device)
    va_logits, va_labels, va_mask = _split_logits(va_logits, val_graph)
    te_logits, te_labels, te_mask = _split_logits(te_logits, test_graph)
    y_tr, z_tr = _masked_numpy(tr_logits, train_graph.labels, train_graph.pair_mask)
    y_va, z_va = _masked_numpy(va_logits, va_labels, va_mask)
    y_te, z_te = _masked_numpy(te_logits, te_labels, te_mask)
    test_metrics = compute_metrics(y_te, z_te)
    rest = [row["auprc"] for row in test_metrics["per_class"][1:] if row["auprc"] is not None]
    dial = test_metrics["per_class"][0]["auprc"] or 0.0
    mean_rest = float(np.mean(rest)) if rest else 0.0
    if dial > 5 * mean_rest:
        dominance = "Gains, if any, remain **dialysis-dominated**: dialysis AUPRC still dwarfs the other 24 classes."
    else:
        dominance = "AUPRC is **more distributed** than Vanilla BCE: dialysis no longer accounts for nearly all ranking mass."

    payload = {
        "seed": seed,
        "split": config["training"]["patient_split"],
        "epochs": epochs,
        "experiment": experiment_name,
        "lr": config["training"]["lr"],
        "param_count": param_count,
        "device": str(device),
        "hardware": f"{platform.processor()} | cuda={torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        "runtime_sec": runtime,
        "best_epoch": best_epoch,
        "checkpoint": str(best_path),
        "early_stopping": (
            f"patience={early_stop_patience} on unweighted val BCE; stopped_early={stopped_early}"
            if early_stop_patience is not None
            else "off"
        ),
        "stopped_early": stopped_early,
        "history": history,
        "pos_weights": pos_weight.detach().cpu().tolist(),
        "train_pos": [int(x) for x in train_pos],
        "train_metrics": compute_metrics(y_tr, z_tr),
        "val_metrics": compute_metrics(y_va, z_va),
        "test_metrics": test_metrics,
        "numerical_notes": "None recorded.",
        "dominance_note": dominance,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_weighted_bce_report(report_path, payload)
    (checkpoint_dir / history_name).write_text(json.dumps(history, indent=2), encoding="utf-8")
    if compare_weighted_bce_20:
        _append_convergence_comparison(report_path, payload)
    return payload
