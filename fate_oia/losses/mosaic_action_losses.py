from __future__ import annotations

import torch
from torch.nn import functional as F


def action_asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Clipped asymmetric multi-label loss from probabilities."""
    if logits.shape != targets.shape:
        raise ValueError("action logits and targets must have matching shapes")
    probability = torch.sigmoid(logits)
    negative_probability = (1.0 - probability + clip).clamp(max=1.0)
    positive = targets * torch.log(probability.clamp_min(eps))
    negative = (1.0 - targets) * torch.log(negative_probability.clamp_min(eps))
    if gamma_pos > 0:
        positive = positive * (1.0 - probability).pow(gamma_pos)
    if gamma_neg > 0:
        negative = negative * (1.0 - negative_probability).pow(gamma_neg)
    return -(positive + negative).mean()


def action_cardinality_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError("action logits and targets must have matching shapes")
    predicted_cardinality = torch.sigmoid(logits).sum(dim=-1)
    target_cardinality = targets.sum(dim=-1)
    return F.smooth_l1_loss(predicted_cardinality, target_cardinality)


def build_mosaic_action_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    rank_loss: torch.Tensor | None = None,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    rank_weight: float = 0.10,
    cardinality_weight: float = 0.02,
) -> dict[str, torch.Tensor]:
    if rank_weight < 0 or cardinality_weight < 0:
        raise ValueError("action loss weights must be non-negative")
    action_asl = action_asymmetric_loss(
        logits,
        targets,
        gamma_pos=gamma_pos,
        gamma_neg=gamma_neg,
        clip=clip,
    )
    if rank_loss is None:
        rank_loss = logits.sum() * 0.0
    if rank_loss.ndim != 0:
        raise ValueError("action rank loss must be scalar")
    cardinality = action_cardinality_loss(logits, targets)
    return {
        "loss_action_asl": action_asl,
        "loss_action_rank": rank_loss,
        "loss_action_cardinality": cardinality,
        "loss_action_total": action_asl + rank_weight * rank_loss + cardinality_weight * cardinality,
    }
