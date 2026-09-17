from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from src.data.processed_bundle import ProcessedBundle
from src.evaluation.metrics import compute_metrics
from src.graph.splits import build_eval_graph, build_split_graph, patient_encounter_index
from src.models.mingle import MingleModel
from src.training.focal import FOCAL_ALPHA, FOCAL_GAMMA, masked_binary_focal_loss
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
from src.training.train_weighted_bce import VANILLA_BCE_TEST
from src.training.validate_architecture import masked_bce_with_logits

# Weighted BCE 50-epoch converged test numbers (do not overwrite 4A artifacts).
WEIGHTED_BCE_50_TEST = {
    "bce": 0.346839,
    "micro_f1": 0.1172,
    "macro_f1": 0.1374,
    "macro_auroc": 0.9103792169007596,
    "micro_auprc": 0.3390088719738635,
    "macro_auprc": 0.27067037987480197,
    "n_positive_predictions": 94679,
    "pct_positive_predictions": 17.3333,
    "best_epoch": 49,
    "per_class_auprc": [
        0.8437717856964287,
        0.21082238851478857,
        0.5264041395103363,
        0.11564343595096302,
        0.6609869533233595,
        0.12211729943370254,
        0.04037341758971558,
        0.5818365385762887,
        0.5737559488119758,
        0.6637754183547397,
        0.031760757565367434,
        0.03362465868610937,
        0.022926853376457182,
        0.07294234962440331,
        0.017564023881180436,
        0.11608629025208657,
        0.6579245636294139,
        0.04239032920129909,
        0.8408974006846077,
        0.02224671697125173,
        0.1484866111683669,
        0.07864233260141743,
        0.17173650484500017,
        0.14439891169509148,
        0.025643866925697804,
    ],
    "per_class_auroc": [
        0.9924800011512132,
        0.7982392178600527,
        0.9608717205204171,
        0.8934710647887972,
        0.981437138697075,
        0.939970035633301,
        0.8527989114380591,
        0.9578098609288206,
        0.9558215010141988,
        0.9616077741055687,
        0.8497875954313516,
        0.8887527376665801,
        0.8367758231521623,
        0.9578239576263292,
        0.7651585545876911,
        0.8885551718102056,
        0.9717801031101321,
        0.8947921609445235,
        0.9974177599229274,
        0.847056758096497,
        0.9226220250100847,
        0.8932931966848966,
        0.9864911409687529,
        0.8983918767969055,
        0.8662743345724436,
    ],
    "per_class_f1": [
        0.7035, 0.2912, 0.4774, 0.1209, 0.3332, 0.1230, 0.0618, 0.1056, 0.1057, 0.0843,
        0.0500, 0.0435, 0.0354, 0.0365, 0.0255, 0.0530, 0.0542, 0.0306, 0.5034, 0.0301,
        0.0356, 0.0270, 0.0449, 0.0342, 0.0255,
    ],
}


def write_focal_loss_report(path: Path, payload: dict) -> None:
    s3 = VANILLA_BCE_TEST
    s4 = WEIGHTED_BCE_50_TEST
    t = payload["test_metrics"]
    v = payload["val_metrics"]
    tr = payload["train_metrics"]
    rest = [row["auprc"] for row in t["per_class"][1:] if row["auprc"] is not None]
    dialysis_auprc = t["per_class"][0]["auprc"]
    n_gt_s3 = sum(
        1
        for i, row in enumerate(t["per_class"])
        if row["auprc"] is not None and row["auprc"] > s3["per_class_auprc"][i]
    )
    n_gt_s4 = sum(
        1
        for i, row in enumerate(t["per_class"])
        if row["auprc"] is not None and row["auprc"] > s4["per_class_auprc"][i]
    )
    n_lt_s4 = sum(
        1
        for i, row in enumerate(t["per_class"])
        if row["auprc"] is not None and row["auprc"] < s4["per_class_auprc"][i]
    )
    d_macro_auprc_s3 = t["macro_auprc"] - s3["macro_auprc"]
    d_macro_auprc_s4 = t["macro_auprc"] - s4["macro_auprc"]
    d_micro_auprc_s3 = t["micro_auprc"] - s3["micro_auprc"]
    d_micro_auprc_s4 = t["micro_auprc"] - s4["micro_auprc"]
    d_macro_auroc_s3 = t["macro_auroc"] - s3["macro_auroc"]
    d_macro_auroc_s4 = t["macro_auroc"] - s4["macro_auroc"]

    better_than_4a = (
        t["macro_auprc"] > s4["macro_auprc"] and t["micro_auprc"] > s4["micro_auprc"]
    )
    worse_than_4a = (
        t["macro_auprc"] < s4["macro_auprc"] and t["micro_auprc"] < s4["micro_auprc"]
    )
    if better_than_4a:
        conclusion = (
            "Focal Loss **improves** over Weighted BCE on both macro-AUPRC and micro-AUPRC."
        )
    elif worse_than_4a:
        conclusion = (
            "Focal Loss **does not improve** over Weighted BCE: both macro-AUPRC and "
            "micro-AUPRC are lower than Weighted BCE."
        )
    else:
        conclusion = (
            "Focal Loss **does not clearly improve** over Weighted BCE: ranking metrics "
            "move in mixed directions (see the comparison table). Do not treat Focal Loss "
            "as better than Weighted BCE on this evidence."
        )

    lines = [
        "# Focal loss",
        "",
        "## 1. Objective",
        "",
        "Test whether **Focal Loss alone** improves next-visit 25-label ranking over "
        "Vanilla BCE vanilla BCE and Weighted BCE class-weighted BCE, with every other experimental "
        "condition held fixed.",
        "",
        "## 2. Exact experimental setup",
        "",
        f"- seed: `{payload['seed']}`",
        f"- split: `{payload['split']}`",
        f"- epochs (max): `{payload['epochs']}`",
        f"- optimizer: Adam, lr `{payload['lr']}` (no weight decay)",
        f"- d=48, h=4, L=2, 25 classes, PMA + PairNorm + MLP_1/2 + self-loops + JK-CONCAT",
        f"- parameter count: `{payload['param_count']}`",
        f"- device: `{payload['device']}`",
        f"- hardware: `{payload['hardware']}`",
        f"- runtime_sec: `{payload['runtime_sec']:.1f}`",
        f"- best epoch: `{payload['best_epoch']}` (lowest **unweighted** val BCE)",
        f"- checkpoint: `{payload['checkpoint']}`",
        f"- early stopping: `{payload.get('early_stopping', 'off')}`",
        f"- loss: binary focal, gamma=`{payload['focal_gamma']}`, alpha=`{payload['focal_alpha']}`",
        f"- class-weighted BCE: `{payload['use_class_weights']}`",
        "",
        "Checkpoint selection matches Weighted BCE: **unweighted validation BCE**, not focal loss "
        "and not test metrics. F1 uses threshold 0.5. Full-graph training (no mini-batches).",
        "",
        "## 3. What was changed from Weighted BCE",
        "",
        "The **training loss only**. Weighted BCE used `BCEWithLogitsLoss(pos_weight=n_neg/n_pos)` "
        "from train pairs. Focal loss uses binary focal loss on the same masked cells, with "
        f"**gamma={payload['focal_gamma']}** (Lin et al. 2017 default) and **alpha={payload['focal_alpha']}** "
        "(no class-balancing term). These values were chosen **a priori**, not tuned on val or test.",
        "",
        "Unchanged: model, graph construction, embeddings, labels, patient split, seed, Adam lr, "
        "full-graph setup, evaluation pipeline, checkpoint rule, max 50 epochs, early-stop patience 10 "
        "on unweighted val BCE.",
        "",
        "## 4. Training / convergence",
        "",
        "| Epoch | Train focal | Train unweighted BCE | Val focal | Val unweighted BCE |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["history"]:
        lines.append(
            f"| {row['epoch']} | {row['train_focal']:.6f} | {row['train_bce']:.6f} | "
            f"{row['val_focal']:.6f} | {row['val_bce']:.6f} |"
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
    lines.extend(
        [
            "",
            "## 5. Test results",
            "",
            f"- n examples: `{t['n_examples']}`",
            f"- unweighted BCE: `{t['bce']:.6f}`",
            f"- micro-AUPRC: `{t['micro_auprc']}`",
            f"- macro-AUPRC: `{t['macro_auprc']}`",
            f"- macro-AUROC: `{t['macro_auroc']}`",
            f"- micro-F1 @0.5: `{t['micro_f1']:.4f}`",
            f"- macro-F1 @0.5: `{t['macro_f1']:.4f}`",
            f"- positive predictions @0.5: `{t.get('n_positive_predictions', 'NA')}` "
            f"({t.get('pct_positive_predictions', 0):.4f}% of cells)",
            "",
            "| Rank | Code | Name | AUROC | AUPRC | F1@0.5 | P | R | Support |",
            "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in t["per_class"]:
        lines.append(
            f"| {row['rank']} | `{row['code']}` | {row['name']} | {row['auroc']} | {row['auprc']} | "
            f"{row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['support']} |"
        )

    lines.extend(
        [
            "",
            "## 6. Vanilla BCE vs weighted BCE vs focal loss (test)",
            "",
            "| Metric | Vanilla BCE Vanilla BCE | Weighted BCE Weighted BCE (50-ep) | Focal loss Focal | Δ vs Vanilla BCE | Δ vs Weighted BCE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
            f"| unweighted BCE | {s3['bce']:.6f} | {s4['bce']:.6f} | {t['bce']:.6f} | {t['bce']-s3['bce']:.6f} | {t['bce']-s4['bce']:.6f} |",
            f"| micro-AUPRC | {s3['micro_auprc']:.6f} | {s4['micro_auprc']:.6f} | {t['micro_auprc']:.6f} | {d_micro_auprc_s3:.6f} | {d_micro_auprc_s4:.6f} |",
            f"| macro-AUPRC | {s3['macro_auprc']:.6f} | {s4['macro_auprc']:.6f} | {t['macro_auprc']:.6f} | {d_macro_auprc_s3:.6f} | {d_macro_auprc_s4:.6f} |",
            f"| macro-AUROC | {s3['macro_auroc']:.6f} | {s4['macro_auroc']:.6f} | {t['macro_auroc']:.6f} | {d_macro_auroc_s3:.6f} | {d_macro_auroc_s4:.6f} |",
            f"| micro-F1 @0.5 | {s3['micro_f1']:.4f} | {s4['micro_f1']:.4f} | {t['micro_f1']:.4f} | {t['micro_f1']-s3['micro_f1']:.4f} | {t['micro_f1']-s4['micro_f1']:.4f} |",
            f"| macro-F1 @0.5 | {s3['macro_f1']:.4f} | {s4['macro_f1']:.4f} | {t['macro_f1']:.4f} | {t['macro_f1']-s3['macro_f1']:.4f} | {t['macro_f1']-s4['macro_f1']:.4f} |",
            f"| # pos preds @0.5 | {s3['n_positive_predictions']} | {s4['n_positive_predictions']} | {t.get('n_positive_predictions', 0)} | | |",
            "",
            f"- change in macro-AUPRC vs Vanilla BCE: `{d_macro_auprc_s3:.6f}`",
            f"- change in macro-AUPRC vs Weighted BCE: `{d_macro_auprc_s4:.6f}`",
            f"- change in micro-AUPRC vs Vanilla BCE: `{d_micro_auprc_s3:.6f}`",
            f"- change in micro-AUPRC vs Weighted BCE: `{d_micro_auprc_s4:.6f}`",
            f"- change in macro-AUROC vs Vanilla BCE: `{d_macro_auroc_s3:.6f}`",
            f"- change in macro-AUROC vs Weighted BCE: `{d_macro_auroc_s4:.6f}`",
            f"- classes with AUPRC > Vanilla BCE: `{n_gt_s3}/25`",
            f"- classes with AUPRC > Weighted BCE: `{n_gt_s4}/25`",
            f"- classes with AUPRC < Weighted BCE: `{n_lt_s4}/25`",
            "",
            "## 7. Per-class analysis",
            "",
            "| Rank | Name | S3 AUPRC | 4A AUPRC | 4B AUPRC | vs Vanilla BCE | vs Weighted BCE |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for i, row in enumerate(t["per_class"]):
        a3 = s3["per_class_auprc"][i]
        a4 = s4["per_class_auprc"][i]
        a5 = row["auprc"] or 0.0
        lines.append(
            f"| {row['rank']} | {row['name']} | {a3:.4f} | {a4:.4f} | {a5:.4f} | {a5-a3:.4f} | {a5-a4:.4f} |"
        )
    mean_rest = float(np.mean(rest)) if rest else float("nan")
    lines.extend(
        [
            "",
            f"- dialysis test AUPRC: `{dialysis_auprc}`",
            f"- mean AUPRC of other 24 classes: `{mean_rest}`",
            "",
            "## 8. Interpretation",
            "",
            payload["dominance_note"],
            "",
            "Checkpointing uses unweighted BCE, which is a poor proxy for ranking under sparsity "
            "(Vanilla BCE already showed this). Differences vs Weighted BCE are therefore also affected by "
            "where the unweighted-BCE minimum lands, not only by the training loss.",
            "",
            "## 9. Limitations",
            "",
            "- Single seed, one Coherent dump, CPU/GPU hardware may differ from prior runs.",
            "- gamma/alpha were not swept; only the Lin et al. focusing default is tested.",
            "- Focal Loss is not combined with pos_weight; a joint recipe is a different experiment.",
            "- F1@0.5 remains a harsh operating point under class imbalance.",
            "- vanilla BCE / weighted BCE checkpoints were not retrained; comparison uses stored test metrics.",
            "",
            "## 10. Conclusion",
            "",
            conclusion,
            "",
            "## Numerical issues",
            "",
            payload.get("numerical_notes", "None recorded."),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def train_focal_loss(
    config: dict,
    bundle: ProcessedBundle,
    report_path: Path,
    *,
    epochs: int = 50,
    checkpoint_name: str = "focal_loss_best.pt",
    history_name: str = "focal_loss_history.json",
    early_stop_patience: int = 10,
) -> dict:
    if bundle.source != "processed":
        raise RuntimeError("Refusing Focal loss on synthetic data.")
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
        f"Focal loss focal loss on {device}. gamma={FOCAL_GAMMA} alpha={FOCAL_ALPHA}. "
        "No class-weighted BCE. Frozen architecture.",
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
        train_focal = masked_binary_focal_loss(
            train_logits, train_graph.labels, train_graph.pair_mask
        )
        _check_finite(train_focal, epoch)
        train_focal.backward()
        optimizer.step()
        train_fl_item = float(train_focal.item())
        with torch.no_grad():
            train_u = masked_bce_with_logits(train_logits, train_graph.labels, train_graph.pair_mask)
            train_u_item = float(train_u.item())
        del train_logits, train_focal, train_u
        if device.type == "cuda":
            torch.cuda.empty_cache()
        model.eval()
        val_logits = _eval_forward(model, bundle, val_graph, device)
        val_logits, val_labels, val_mask = _split_logits(val_logits, val_graph)
        val_u = masked_bce_with_logits(val_logits, val_labels, val_mask)
        val_focal = masked_binary_focal_loss(val_logits, val_labels, val_mask)
        _check_finite(val_u, epoch)
        history.append(
            {
                "epoch": epoch,
                "train_focal": train_fl_item,
                "train_bce": train_u_item,
                "val_focal": float(val_focal.item()),
                "val_bce": float(val_u.item()),
            }
        )
        print(
            f"epoch {epoch}/{epochs} train_fl={train_fl_item:.6f} train_u={train_u_item:.6f} "
            f"val_fl={val_focal.item():.6f} val_u={val_u.item():.6f}",
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
                    "focal_gamma": FOCAL_GAMMA,
                    "focal_alpha": FOCAL_ALPHA,
                    "experiment": "focal_loss",
                },
                best_path,
            )
        else:
            stale += 1
            if early_stop_patience is not None and stale >= early_stop_patience:
                stopped_early = True
                print(
                    f"Early stop at epoch {epoch}: unweighted val BCE did not improve for "
                    f"{early_stop_patience} epochs.",
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
        dominance = (
            "Gains, if any, remain **dialysis-dominated**: dialysis AUPRC still dwarfs the other 24 classes."
        )
    else:
        dominance = (
            "AUPRC is **more distributed** than Vanilla BCE: dialysis no longer accounts for nearly all ranking mass."
        )

    payload = {
        "seed": seed,
        "split": config["training"]["patient_split"],
        "epochs": epochs,
        "experiment": "focal_loss",
        "lr": config["training"]["lr"],
        "param_count": param_count,
        "device": str(device),
        "hardware": f"{platform.processor()} | cuda={torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        "runtime_sec": runtime,
        "best_epoch": best_epoch,
        "checkpoint": str(best_path),
        "early_stopping": (
            f"patience={early_stop_patience} on unweighted val BCE; stopped_early={stopped_early}"
        ),
        "stopped_early": stopped_early,
        "history": history,
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha": FOCAL_ALPHA,
        "use_class_weights": False,
        "train_metrics": compute_metrics(y_tr, z_tr),
        "val_metrics": compute_metrics(y_va, z_va),
        "test_metrics": test_metrics,
        "numerical_notes": "None recorded.",
        "dominance_note": dominance,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_focal_loss_report(report_path, payload)
    (checkpoint_dir / history_name).write_text(json.dumps(history, indent=2), encoding="utf-8")
    return payload
