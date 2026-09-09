from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from src.data.stage3_labels import FROZEN_TOP25


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))


def score_threshold(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    y = y_true.astype(np.int32)
    pred = (probs >= threshold).astype(np.int32)
    n_pos = int(pred.sum())
    n_cells = int(pred.size)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y, pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y, pred, average="macro", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y, pred, average=None, zero_division=0
    )
    per_class = []
    for i, (code, name) in enumerate(FROZEN_TOP25):
        per_class.append(
            {
                "rank": i + 1,
                "code": code,
                "name": name,
                "f1": float(per_f1[i]),
                "precision": float(per_p[i]),
                "recall": float(per_r[i]),
                "support": int(support[i]),
            }
        )
    return {
        "threshold": threshold,
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "n_positive_predictions": n_pos,
        "pct_positive_predictions": 100.0 * n_pos / n_cells if n_cells else 0.0,
        "per_class": per_class,
    }


def select_threshold(val_rows: list[dict]) -> dict:
    """Pick operating point on validation only: max micro-F1, then macro-F1, then larger threshold."""
    return max(val_rows, key=lambda row: (row["micro_f1"], row["macro_f1"], row["threshold"]))
