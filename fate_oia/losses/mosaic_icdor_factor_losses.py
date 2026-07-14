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


def factor_presence_visibility_losses(
    factor_presence_logits: torch.Tensor,
    factor_visibility_logits: torch.Tensor,
    presence_targets: torch.Tensor,
    visibility_targets: torch.Tensor,
    presence_known_mask: torch.Tensor,
    visibility_known_mask: torch.Tensor,
    weak_negative_mask: torch.Tensor | None = None,
    weak_negative_weight: float = 0.30,
) -> dict[str, torch.Tensor]:
    """Supervise only observed/geometry-known factor states; unknown is ignored."""
    loss_presence = _masked_bce(factor_presence_logits, presence_targets, presence_known_mask)
    loss_visibility = _masked_bce(factor_visibility_logits, visibility_targets, visibility_known_mask)
    if weak_negative_mask is None:
        loss_weak_negative = factor_presence_logits.sum() * 0.0
    else:
        if weak_negative_mask.shape != factor_presence_logits.shape or not 0.0 <= weak_negative_weight <= 1.0:
            raise ValueError("IC-DOR weak-negative supervision is invalid")
        weak = weak_negative_mask.to(dtype=factor_presence_logits.dtype)
        raw = F.softplus(factor_presence_logits)
        loss_weak_negative = (raw * weak).sum() / weak.sum().clamp_min(1.0)
    return {
        "loss_factor_presence": loss_presence,
        "loss_factor_visibility": loss_visibility,
        "loss_factor_weak_negative": loss_weak_negative,
        "loss_factor_total": loss_presence + loss_visibility + weak_negative_weight * loss_weak_negative,
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
