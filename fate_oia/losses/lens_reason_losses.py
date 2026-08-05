from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits


def reason_asl_loss(logits: torch.Tensor, targets: torch.Tensor, shared_weight: torch.Tensor | None = None) -> torch.Tensor:
    raw = asymmetric_loss_with_logits(logits, targets.float(), gamma_neg=4, gamma_pos=0, clip=0.05, reduction="none")
    if shared_weight is None:
        return raw.mean()
    return (raw * shared_weight).sum() / shared_weight.sum().clamp_min(1.0)


def reason_rank_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    positive = logits.masked_fill(targets <= 0.5, float("inf")).min(-1).values
    negative = logits.masked_fill(targets > 0.5, float("-inf")).max(-1).values
    valid = torch.isfinite(positive) & torch.isfinite(negative)
    return F.relu(0.1 - positive[valid] + negative[valid]).mean() if valid.any() else logits.sum() * 0.0


def reason_soft_f1_loss(logits: torch.Tensor, targets: torch.Tensor, negative_weight: torch.Tensor | None = None) -> torch.Tensor:
    probs = logits.sigmoid(); target = targets.float()
    neg = torch.ones_like(target) if negative_weight is None else negative_weight
    tp = (probs * target).sum(0); fp = (probs * (1 - target) * neg).sum(0); fn = ((1 - probs) * target).sum(0)
    return 1.0 - ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean()
