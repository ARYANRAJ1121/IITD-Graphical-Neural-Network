from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.next_visit_labels import FROZEN_TOP25
from src.evaluation.thresholds import sigmoid


def _safe_auroc(y: np.ndarray, scores: np.ndarray) -> float | None:
    if y.min() == y.max():
        return None
    return float(roc_auc_score(y, scores))


def _safe_auprc(y: np.ndarray, scores: np.ndarray) -> float | None:
    if y.sum() == 0:
        return None
    return float(average_precision_score(y, scores))


def precision_recall_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    n = int(y.shape[0])
    kk = min(k, n)
    if kk == 0:
        return 0.0, 0.0
    order = np.argsort(-scores)[:kk]
    hits = float(y[order].sum())
    n_pos = float(y.sum())
    precision = hits / kk
    recall = (hits / n_pos) if n_pos > 0 else 0.0
    return precision, recall


def class_row(
    index: int,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    p_test: np.ndarray,
) -> dict:
    code, name = FROZEN_TOP25[index]
    yt, yv, ye = y_train[:, index], y_val[:, index], y_test[:, index]
    scores = p_test[:, index]
    pos_mask = ye == 1
    neg_mask = ye == 0
    n_pos = int(ye.sum())
    prev_te = float(ye.mean()) if ye.size else 0.0
    auroc = _safe_auroc(ye, scores)
    auprc = _safe_auprc(ye, scores)
    lift = (auprc / prev_te) if (auprc is not None and prev_te > 0) else None
    p1, r1 = precision_recall_at_k(ye, scores, 1)
    p3, r3 = precision_recall_at_k(ye, scores, 3)
    p5, r5 = precision_recall_at_k(ye, scores, 5)
    return {
        "rank": index + 1,
        "code": code,
        "name": name,
        "train_prev": float(yt.mean()) if yt.size else 0.0,
        "val_prev": float(yv.mean()) if yv.size else 0.0,
        "test_prev": prev_te,
        "train_pos": int(yt.sum()),
        "val_pos": int(yv.sum()),
        "test_pos": n_pos,
        "auroc": auroc,
        "auprc": auprc,
        "auprc_lift": lift,
        "mean_p_pos": float(scores[pos_mask].mean()) if n_pos else None,
        "mean_p_neg": float(scores[neg_mask].mean()) if int(neg_mask.sum()) else None,
        "p_at_1": p1,
        "r_at_1": r1,
        "p_at_3": p3,
        "r_at_3": r3,
        "p_at_5": p5,
        "r_at_5": r5,
    }


def macro_mean(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return None
    return float(np.mean(values))
