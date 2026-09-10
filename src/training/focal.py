from __future__ import annotations

import torch
from torch.nn import functional as F


# Lin et al., "Focal Loss for Dense Object Detection" (2017). Chosen a priori.
# gamma=2 is the paper default. alpha=1 means no class-balancing term, so this
# experiment is focusing-only and is not mixed with Weighted BCE pos_weight.
FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 1.0


def masked_binary_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pair_mask: torch.Tensor,
    *,
    gamma: float = FOCAL_GAMMA,
    alpha: float = FOCAL_ALPHA,
) -> torch.Tensor:
    """Mean binary focal loss over masked (visit, class) pairs.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    with p_t = p if y=1 else 1-p, p = sigmoid(z).
    When alpha=1, alpha_t is identically 1 (no positive/negative reweighting).
    Reduction matches BCEWithLogitsLoss(mean) over the same masked cells.
    """
    z = logits[pair_mask]
    y = labels[pair_mask]
    bce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    p = torch.sigmoid(z)
    p_t = p * y + (1.0 - p) * (1.0 - y)
    modulating = (1.0 - p_t).clamp(min=0.0, max=1.0).pow(gamma)
    if alpha == 1.0:
        return (modulating * bce).mean()
    alpha_t = alpha * y + (1.0 - alpha) * (1.0 - y)
    return (alpha_t * modulating * bce).mean()
