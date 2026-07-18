from __future__ import annotations

import torch
from torch.nn import functional as F


def _masked_bce(logits: torch.Tensor, targets: torch.Tensor, known_mask: torch.Tensor) -> torch.Tensor:
    if logits.shape != targets.shape or logits.shape != known_mask.shape:
        raise ValueError("IC-DOR factor masked BCE requires matching tensors")
    known = known_mask.to(dtype=logits.dtype)
    denominator = known.sum()
    if denominator.item() == 0:
        return logits.sum() * 0.0
    value = F.binary_cross_entropy_with_logits(logits, targets.to(dtype=logits.dtype), reduction="none")
    return (value * known).sum() / denominator


def factor_positive_anchor_loss(
    factor_presence_logits: torch.Tensor,
    supervision: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply weighted positive-only reason anchors to the factor-presence lane."""
    positive_anchor_mask = supervision.get("positive_anchor_mask")
    positive_anchor_weight = supervision.get("positive_anchor_weight")
    if not isinstance(positive_anchor_mask, torch.Tensor) or not isinstance(positive_anchor_weight, torch.Tensor):
        raise ValueError("factor anchor supervision requires mask and weight tensors")
    if factor_presence_logits.shape != positive_anchor_mask.shape or factor_presence_logits.shape != positive_anchor_weight.shape:
        raise ValueError("factor anchor supervision must match presence logits")
    weights = positive_anchor_mask.to(dtype=factor_presence_logits.dtype) * positive_anchor_weight.to(
        dtype=factor_presence_logits.dtype
    )
    denominator = weights.sum()
    if denominator.item() == 0:
        return factor_presence_logits.sum() * 0.0
    return (F.softplus(-factor_presence_logits) * weights).sum() / denominator


def factor_selective_contrastive_loss(
    factor_features: torch.Tensor,
    positive_mask: torch.Tensor,
    reliable_negative_mask: torch.Tensor,
    *,
    negative_margin: float = 0.10,
) -> torch.Tensor:
    """Separate each factor's grounded positives from reliable negatives.

    The reduction is factor-balanced: a frequent object factor cannot drown
    out a sparse lane or traffic-control factor. Unknown and weak-negative
    observations are deliberately excluded from this visual-only objective.
    """
    if factor_features.ndim != 3 or positive_mask.shape != factor_features.shape[:2] or reliable_negative_mask.shape != factor_features.shape[:2]:
        raise ValueError("IC-DOR selective contrastive loss expects [B,F,D] features and [B,F] masks")
    if positive_mask.dtype != torch.bool or reliable_negative_mask.dtype != torch.bool:
        raise ValueError("IC-DOR selective contrastive masks must be boolean")
    if not 0.0 <= float(negative_margin) < 1.0:
        raise ValueError("IC-DOR selective contrastive margin must be in [0,1)")
    normalized = F.normalize(factor_features, dim=-1, eps=1e-6)
    losses: list[torch.Tensor] = []
    for factor_id in range(normalized.shape[1]):
        positives = normalized[positive_mask[:, factor_id], factor_id]
        negatives = normalized[reliable_negative_mask[:, factor_id], factor_id]
        if positives.numel() == 0 or negatives.numel() == 0:
            continue
        prototype = F.normalize(positives.mean(dim=0, keepdim=True), dim=-1, eps=1e-6)
        positive_compactness = (1.0 - (positives * prototype).sum(dim=-1)).mean()
        negative_separation = F.relu((negatives * prototype).sum(dim=-1) - float(negative_margin)).mean()
        losses.append(positive_compactness + negative_separation)
    return torch.stack(losses).mean() if losses else factor_features.sum() * 0.0


def factor_presence_visibility_losses(
    factor_presence_logits: torch.Tensor,
    factor_visibility_logits: torch.Tensor,
    presence_targets: torch.Tensor,
    visibility_targets: torch.Tensor,
    presence_known_mask: torch.Tensor,
    visibility_known_mask: torch.Tensor,
    weak_negative_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Supervise only observed/geometry-known factor states; unknown is ignored."""
    loss_presence = _masked_bce(factor_presence_logits, presence_targets, presence_known_mask)
    loss_visibility = _masked_bce(factor_visibility_logits, visibility_targets, visibility_known_mask)
    if weak_negative_mask is None:
        loss_weak_negative = factor_presence_logits.sum() * 0.0
    else:
        if weak_negative_mask.shape != factor_presence_logits.shape:
            raise ValueError("IC-DOR weak-negative supervision is invalid")
        weak = weak_negative_mask.to(dtype=factor_presence_logits.dtype)
        raw = F.softplus(factor_presence_logits)
        loss_weak_negative = (raw * weak).sum() / weak.sum().clamp_min(1.0)
    return {
        "loss_factor_presence": loss_presence,
        "loss_factor_visibility": loss_visibility,
        "loss_factor_weak_negative": loss_weak_negative,
        # The exact CREDO factor objective is assembled by the trainer as
        # presence + visibility + geometry + selected contrast + view/flip +
        # prototype. Keep weak negatives visible for audit without silently
        # adding a plan-external optimisation term to this aggregate.
        "loss_factor_total": loss_presence + loss_visibility,
    }


def factor_view_consistency_loss(
    first_presence: torch.Tensor,
    second_presence_restored: torch.Tensor,
    first_visibility: torch.Tensor,
    second_visibility_restored: torch.Tensor,
    first_masks: torch.Tensor,
    second_masks_restored: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if first_presence.shape != second_presence_restored.shape or first_visibility.shape != second_visibility_restored.shape:
        raise ValueError("IC-DOR view consistency requires aligned factor probabilities")
    if first_masks.shape != second_masks_restored.shape or first_masks.shape[:2] != first_presence.shape:
        raise ValueError("IC-DOR view consistency requires aligned factor masks")
    probability = F.smooth_l1_loss(first_presence, second_presence_restored) + F.smooth_l1_loss(
        first_visibility, second_visibility_restored
    )
    mask = F.smooth_l1_loss(first_masks, second_masks_restored)
    return {"loss_factor_view_probability": probability, "loss_factor_flip_equivariance": mask, "loss_factor_view_total": probability + mask}


def factor_prototype_regularization(
    prototype_weights: torch.Tensor,
    prototypes: torch.Tensor,
    valid_mask: torch.Tensor,
    prior_scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if prototype_weights.ndim != 3 or prototypes.ndim != 3 or valid_mask.shape != prototypes.shape[:2]:
        raise ValueError("IC-DOR prototype regularization shapes are invalid")
    if prototype_weights.shape[1:] != valid_mask.shape or prior_scale.shape != (valid_mask.shape[0],):
        raise ValueError("IC-DOR prototype weights/prior scale do not match factor ontology")
    valid = valid_mask.to(dtype=prototype_weights.dtype)
    occupancy = prototype_weights.mean(0)
    target = valid / valid.sum(-1, keepdim=True).clamp_min(1.0)
    loss_occupancy = ((occupancy - target).square() * valid).sum() / valid.sum().clamp_min(1.0)
    normalized = F.normalize(prototypes, dim=-1, eps=1e-6)
    cosine = torch.einsum("fkd,fjd->fkj", normalized, normalized)
    pair_mask = valid_mask[:, :, None] & valid_mask[:, None, :]
    pair_mask = pair_mask & ~torch.eye(valid_mask.shape[1], dtype=torch.bool, device=valid_mask.device).unsqueeze(0)
    loss_repulsion = (cosine.clamp_min(0.0).square() * pair_mask).sum() / pair_mask.sum().clamp_min(1)
    loss_prior = prior_scale.square().mean()
    return {"loss_factor_prototype_occupancy": loss_occupancy, "loss_factor_prototype_repulsion": loss_repulsion, "loss_factor_prior_scale": loss_prior}


def factor_geometry_alignment_loss(
    factor_soft_masks: torch.Tensor,
    geometry_masks: torch.Tensor,
    geometry_known_mask: torch.Tensor,
) -> torch.Tensor:
    if factor_soft_masks.shape != geometry_masks.shape or factor_soft_masks.shape[:2] != geometry_known_mask.shape:
        raise ValueError("IC-DOR geometry alignment requires masks [B,F,H,W] and known mask [B,F]")
    known = geometry_known_mask.to(dtype=factor_soft_masks.dtype).unsqueeze(-1).unsqueeze(-1)
    denominator = known.sum() * factor_soft_masks.shape[-1] * factor_soft_masks.shape[-2]
    if denominator.item() == 0:
        return factor_soft_masks.sum() * 0.0
    return ((factor_soft_masks - geometry_masks.to(dtype=factor_soft_masks.dtype)).abs() * known).sum() / denominator


def factor_contradiction_consistency_loss(
    factor_presence_probability: torch.Tensor,
    contradiction_mask: torch.Tensor,
) -> torch.Tensor:
    if contradiction_mask.dtype != torch.bool or contradiction_mask.shape != (factor_presence_probability.shape[1],) * 2:
        raise ValueError("IC-DOR contradiction mask must be [F,F] bool")
    pair_probability = factor_presence_probability.unsqueeze(-1) * factor_presence_probability.unsqueeze(-2)
    count = contradiction_mask.sum()
    return (pair_probability * contradiction_mask.to(dtype=pair_probability.dtype)).sum() / count.clamp_min(1)
