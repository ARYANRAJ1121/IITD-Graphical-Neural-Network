from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.data.next_visit_labels import FROZEN_TOP25


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.min() == y_true.max():
        return None
    return float(roc_auc_score(y_true, y_score))


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, probs: np.ndarray | None = None) -> dict:
    if probs is None:
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
    pred = (probs >= 0.5).astype(np.int32)
    y = y_true.astype(np.int32)
    n, c = y.shape
    prevalence = y.mean(axis=0)

    per_class = []
    prec, rec, f1, support = precision_recall_fscore_support(
        y, pred, average=None, zero_division=0
    )
    for i, (code, name) in enumerate(FROZEN_TOP25):
        auroc = _safe_auroc(y[:, i], probs[:, i])
        try:
            auprc = float(average_precision_score(y[:, i], probs[:, i]))
        except ValueError:
            auprc = None
        per_class.append(
            {
                "rank": i + 1,
                "code": code,
                "name": name,
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "prevalence": float(prevalence[i]),
                "auroc": auroc,
                "auprc": auprc,
            }
        )

    micro_f1 = float(f1_score(y, pred, average="micro", zero_division=0))
    macro_f1 = float(f1_score(y, pred, average="macro", zero_division=0))

    defined_auroc = [row["auroc"] for row in per_class if row["auroc"] is not None]
    defined_auprc = [row["auprc"] for row in per_class if row["auprc"] is not None]
    try:
        micro_auprc = float(average_precision_score(y, probs, average="micro"))
    except ValueError:
        micro_auprc = None
    try:
        macro_auprc = float(average_precision_score(y, probs, average="macro"))
    except ValueError:
        macro_auprc = None

    bce = float(
        -(
            y * np.log(np.clip(probs, 1e-8, 1.0))
            + (1 - y) * np.log(np.clip(1.0 - probs, 1e-8, 1.0))
        ).mean()
    )
    n_pos_pred = int(pred.sum())
    return {
        "n_examples": n,
        "n_classes": c,
        "bce": bce,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "macro_auroc": float(np.mean(defined_auroc)) if defined_auroc else None,
        "micro_auprc": micro_auprc,
        "macro_auprc": macro_auprc,
        "classes_with_auroc": len(defined_auroc),
        "n_positive_predictions": n_pos_pred,
        "pct_positive_predictions": 100.0 * n_pos_pred / pred.size if pred.size else 0.0,
        "per_class": per_class,
    }


def always_negative_baseline(y_true: np.ndarray) -> dict:
    logits = np.full_like(y_true, -1e6, dtype=np.float64)
    return compute_metrics(y_true, logits)


def majority_baseline(y_train: np.ndarray, y_eval: np.ndarray) -> dict:
    majority = (y_train.mean(axis=0) >= 0.5).astype(np.float64)
    row = np.where(majority == 1.0, 1e6, -1e6)
    logits = np.broadcast_to(row, y_eval.shape).copy()
    return compute_metrics(y_eval, logits)
