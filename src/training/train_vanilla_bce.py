from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from src.data.processed_bundle import ProcessedBundle
from src.data.next_visit_labels import FROZEN_TOP25
from src.evaluation.metrics import always_negative_baseline, compute_metrics, majority_baseline
from src.graph.splits import SplitGraph, build_eval_graph, build_split_graph, patient_encounter_index
from src.models.mingle import MingleModel
from src.training.validate_architecture import masked_bce_with_logits


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _to_device(graph: SplitGraph, device: torch.device) -> SplitGraph:
    return SplitGraph(
        name=graph.name,
        original_index=graph.original_index,
        notes=graph.notes.to(device),
        labels=graph.labels.to(device),
        pair_mask=graph.pair_mask.to(device),
        incidence_node=graph.incidence_node.to(device),
        incidence_edge=graph.incidence_edge.to(device),
        num_real=graph.num_real,
        eval_offset=graph.eval_offset,
    )


def _forward(model: MingleModel, bundle: ProcessedBundle, graph: SplitGraph) -> torch.Tensor:
    outputs = model(
        bundle.node_states,
        bundle.concept_semantics,
        graph.notes,
        graph.incidence_node,
        graph.incidence_edge,
    )
    logits = outputs["logits"]
    if logits.size(0) != graph.num_real:
        raise RuntimeError(
            f"Logit/label alignment error: logits={tuple(logits.shape)} num_real={graph.num_real}"
        )
    return logits


def _cuda_mem(prefix: str, device: torch.device) -> None:
    if device.type != "cuda":
        return
    alloc = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    print(f"{prefix} cuda alloc={alloc:.2f}GB reserved={reserved:.2f}GB", flush=True)


def _eval_forward(model: MingleModel, bundle: ProcessedBundle, graph: SplitGraph, device: torch.device) -> torch.Tensor:
    """Val/test graphs stay on CPU so backward is not fighting them for VRAM."""
    if graph.notes.device == device:
        with torch.no_grad():
            return _forward(model, bundle, graph)
    moved = _to_device(graph, device)
    try:
        with torch.no_grad():
            return _forward(model, bundle, moved)
    finally:
        del moved
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _split_logits(logits: torch.Tensor, graph: SplitGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = graph.eval_offset
    return logits[start:], graph.labels[start:], graph.pair_mask[start:]


def _masked_numpy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    return labels[mask].detach().cpu().numpy(), logits[mask].detach().cpu().numpy()


def _check_finite(loss: torch.Tensor, epoch: int) -> None:
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite training loss at epoch {epoch}: {loss.item()}")


def write_training_report(path: Path, payload: dict) -> None:
    lines = [
        "# Vanilla BCE training report",
        "",
        "Baseline replication run. Architecture is frozen. Hyperparameters were not tuned.",
        "Decreasing BCE is not treated as task success.",
        "",
        "## Hyperparameters",
        "",
        f"- seed: `{payload['seed']}`",
        f"- patient split: `{payload['split']}`",
        f"- epochs: `{payload['epochs']}`",
        f"- optimizer: `{payload['optimizer']}`",
        f"- lr: `{payload['lr']}`",
        f"- loss: `BCEWithLogitsLoss` (MINGLE Eq. 4)",
        f"- d: `48`, h: `4`, L: `2`, classes: `25`",
        f"- focal/class weights/oversampling: `false`",
        f"- parameter count: `{payload['param_count']}`",
        f"- device: `{payload['device']}`",
        f"- hardware: `{payload['hardware']}`",
        f"- runtime_sec: `{payload['runtime_sec']:.1f}`",
        f"- best epoch: `{payload['best_epoch']}` (lowest validation BCE)",
        f"- checkpoint: `{payload['checkpoint']}`",
        "",
        "## Epoch losses",
        "",
        "| Epoch | Train BCE | Val BCE |",
        "| ---: | ---: | ---: |",
    ]
    for row in payload["history"]:
        lines.append(f"| {row['epoch']} | {row['train_bce']:.6f} | {row['val_bce']:.6f} |")

    def dump_split(title: str, metrics: dict, extra: str = "") -> None:
        lines.extend(["", f"## {title}", "", extra] if extra else ["", f"## {title}", ""])
        lines.extend(
            [
                f"- n examples: `{metrics['n_examples']}`",
                f"- BCE: `{metrics['bce']:.6f}`",
                f"- micro-F1: `{metrics['micro_f1']:.4f}`",
                f"- macro-F1: `{metrics['macro_f1']:.4f}`",
                f"- macro-AUROC (defined classes): `{metrics['macro_auroc']}` ({metrics['classes_with_auroc']}/25)",
                f"- micro-AUPRC: `{metrics['micro_auprc']}`",
                f"- macro-AUPRC: `{metrics['macro_auprc']}`",
                "",
                "| Rank | Code | Name | Prev | P | R | F1 | AUROC | AUPRC | Support |",
                "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in metrics["per_class"]:
            lines.append(
                f"| {row['rank']} | `{row['code']}` | {row['name']} | {row['prevalence']:.4f} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} | {row['f1']:.4f} | "
                f"{row['auroc']} | {row['auprc']} | {row['support']} |"
            )

    dump_split("Train metrics (best checkpoint)", payload["train_metrics"])
    dump_split("Validation metrics (best checkpoint)", payload["val_metrics"])
    dump_split("Test metrics (best checkpoint)", payload["test_metrics"])
    dump_split("Test always-negative baseline", payload["baseline_neg"])
    dump_split(
        "Test majority baseline (from train labels)",
        payload["baseline_maj"],
        extra="If every class has train prevalence < 0.5, this coincides with always-negative.",
    )

    test = payload["test_metrics"]
    base = payload["baseline_neg"]
    lines.extend(
        [
            "",
            "## Baseline comparison (test)",
            "",
            "| Metric | Model | Always-negative | Δ |",
            "| --- | ---: | ---: | ---: |",
            f"| BCE | {test['bce']:.6f} | {base['bce']:.6f} | {test['bce'] - base['bce']:.6f} |",
            f"| micro-F1 | {test['micro_f1']:.4f} | {base['micro_f1']:.4f} | {test['micro_f1'] - base['micro_f1']:.4f} |",
            f"| macro-F1 | {test['macro_f1']:.4f} | {base['macro_f1']:.4f} | {test['macro_f1'] - base['macro_f1']:.4f} |",
            f"| micro-AUPRC | {test['micro_auprc']} | {base['micro_auprc']} | |",
            f"| macro-AUPRC | {test['macro_auprc']} | {base['macro_auprc']} | |",
            "",
            "## Interpretation",
            "",
            "Do not treat a train/val BCE drop as success. Compare micro-F1, macro-F1, and AUPRC against the always-negative baseline under severe sparsity.",
            "",
            "## Numerical issues",
            "",
            payload.get("numerical_notes", "None recorded."),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def train_vanilla_bce(config: dict, bundle: ProcessedBundle, report_path: Path) -> dict:
    if bundle.source != "processed":
        raise RuntimeError("Refusing to run the baseline experiment on synthetic data.")

    frozen = config["model"]
    if (
        frozen["hidden_dim"] != 48
        or frozen["num_heads"] != 4
        or frozen["num_layers"] != 2
        or frozen["num_classes"] != 25
    ):
        raise RuntimeError("Frozen architecture constants were altered in config.")
    if config["training"].get("use_focal_loss") or config["training"].get("use_class_weights") or config["training"].get("use_oversampling"):
        raise RuntimeError("Imbalance corrections are forbidden for this run.")

    device = _device()
    print(f"Training on {device}. Frozen d=48 h=4 L=2 classes=25 BCE. No imbalance corrections.", flush=True)
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_idx = patient_encounter_index(bundle.patient_ids, bundle.split_ids["train"])
    val_idx = patient_encounter_index(bundle.patient_ids, bundle.split_ids["val"])
    test_idx = patient_encounter_index(bundle.patient_ids, bundle.split_ids["test"])
    if np.intersect1d(train_idx, val_idx).size or np.intersect1d(train_idx, test_idx).size or np.intersect1d(val_idx, test_idx).size:
        raise RuntimeError("Patient split leakage: overlapping encounter indices.")

    train_graph = _to_device(build_split_graph(bundle, train_idx, "train"), device)
    val_graph = build_eval_graph(bundle, train_idx, val_idx, "val")
    test_graph = build_eval_graph(bundle, train_idx, test_idx, "test")

    if int(train_graph.pair_mask.sum()) == 0:
        raise RuntimeError("No training pairs after split.")

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
    best_path = checkpoint_dir / "vanilla_bce_best.pt"

    epochs = int(config["training"]["epochs"])
    history = []
    best_val = float("inf")
    best_epoch = -1
    numerical_notes = "None recorded."
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_logits = _forward(model, bundle, train_graph)
        train_loss = masked_bce_with_logits(train_logits, train_graph.labels, train_graph.pair_mask)
        _check_finite(train_loss, epoch)
        train_loss.backward()
        optimizer.step()
        train_bce = float(train_loss.item())
        del train_logits, train_loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

        model.eval()
        val_logits = _eval_forward(model, bundle, val_graph, device)
        val_logits, val_labels, val_mask = _split_logits(val_logits, val_graph)
        val_loss = masked_bce_with_logits(val_logits, val_labels, val_mask)
        _check_finite(val_loss, epoch)

        history.append(
            {
                "epoch": epoch,
                "train_bce": train_bce,
                "val_bce": float(val_loss.item()),
            }
        )
        print(f"epoch {epoch}/{epochs} train_bce={train_bce:.6f} val_bce={val_loss.item():.6f}", flush=True)

        if float(val_loss.item()) < best_val:
            best_val = float(val_loss.item())
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_bce": best_val,
                    "param_count": param_count,
                    "config": {
                        "d": 48,
                        "h": 4,
                        "L": 2,
                        "classes": 25,
                        "lr": config["training"]["lr"],
                        "seed": seed,
                    },
                },
                best_path,
            )

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

    payload = {
        "seed": seed,
        "split": config["training"]["patient_split"],
        "epochs": epochs,
        "optimizer": "adam",
        "lr": config["training"]["lr"],
        "param_count": param_count,
        "device": str(device),
        "hardware": f"{platform.processor()} | cuda={torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        "runtime_sec": runtime,
        "best_epoch": best_epoch,
        "checkpoint": str(best_path),
        "history": history,
        "train_metrics": compute_metrics(y_tr, z_tr),
        "val_metrics": compute_metrics(y_va, z_va),
        "test_metrics": compute_metrics(y_te, z_te),
        "baseline_neg": always_negative_baseline(y_te),
        "baseline_maj": majority_baseline(y_tr, y_te),
        "numerical_notes": numerical_notes,
        "frozen_classes": [name for _, name in FROZEN_TOP25],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_training_report(report_path, payload)
    (checkpoint_dir / "vanilla_bce_history.json").write_text(json.dumps(payload["history"], indent=2), encoding="utf-8")
    return payload
