from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _masked_mean(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    reliability: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape != valid_mask.shape:
        raise ValueError("masked loss values and valid mask must have matching shapes")
    mask = valid_mask.to(dtype=values.dtype)
    count = (valid_mask > 0).sum().detach()
    weights = mask
    if reliability is not None:
        if reliability.shape != values.shape:
            raise ValueError("source reliability must match masked loss values")
        weights = weights * reliability.to(dtype=values.dtype).clamp(0.0, 1.0)
    denominator = weights.sum()
    loss = (values * weights).sum() / denominator.clamp_min(1e-12)
    loss = torch.where(denominator > 0, loss, _zero(values))
    return loss, count


def _optional_consistency(
    first: torch.Tensor,
    second: torch.Tensor | None,
    valid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if second is None:
        return _zero(first), first.new_zeros((), dtype=torch.long)
    if second.shape != first.shape:
        raise ValueError("consistency tensors must have matching shapes")
    if valid_mask is None:
        valid_mask = torch.ones_like(first, dtype=torch.bool)
    values = F.smooth_l1_loss(first, second, reduction="none")
    return _masked_mean(values, valid_mask)


def _factor_balanced_binary_mean(
    values: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    reliability: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Balance observed positive/negative evidence within each factor.

    Unknown observations remain masked. This only prevents the much more
    frequent confirmed positives from overwhelming available weak negatives.
    """
    if not (values.shape == targets.shape == valid_mask.shape == reliability.shape):
        raise ValueError("balanced factor loss tensors must have matching [B,F] shapes")
    if values.ndim != 2:
        raise ValueError("balanced factor loss expects [B,F] tensors")
    base_weight = valid_mask.to(values.dtype) * reliability.to(values.dtype).clamp(0.0, 1.0)
    positive_weight = base_weight * (targets > 0.5).to(values.dtype)
    negative_weight = base_weight * (targets <= 0.5).to(values.dtype)
    positive_denominator = positive_weight.sum(dim=0)
    negative_denominator = negative_weight.sum(dim=0)
    positive_mean = (values * positive_weight).sum(dim=0) / positive_denominator.clamp_min(1e-12)
    negative_mean = (values * negative_weight).sum(dim=0) / negative_denominator.clamp_min(1e-12)
    positive_available = positive_denominator > 0
    negative_available = negative_denominator > 0
    class_count = positive_available.to(values.dtype) + negative_available.to(values.dtype)
    factor_loss = (
        positive_mean * positive_available.to(values.dtype)
        + negative_mean * negative_available.to(values.dtype)
    ) / class_count.clamp_min(1.0)
    factor_available = class_count > 0
    loss = factor_loss[factor_available].mean() if factor_available.any() else _zero(values)
    count = (valid_mask > 0).sum().detach()
    return loss, count


def build_mosaic_factor_loss(
    predictions: dict[str, Any],
    observations: dict[str, torch.Tensor],
    *,
    geometry_mask_weight: float = 0.10,
    view_consistency_weight: float = 0.05,
    flip_equivariance_weight: float = 0.05,
    prototype_occupancy_weight: float = 0.02,
    prototype_repulsion_weight: float = 0.01,
    prior_scale_weight: float = 0.01,
    contradiction_weight: float = 0.02,
) -> dict[str, torch.Tensor]:
    weights = (
        geometry_mask_weight,
        view_consistency_weight,
        flip_equivariance_weight,
        prototype_occupancy_weight,
        prototype_repulsion_weight,
        prior_scale_weight,
        contradiction_weight,
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("factor loss weights must be non-negative")
    required_predictions = {
        "factor_presence_logits",
        "factor_visibility_logits",
        "factor_soft_masks",
        "factor_presence_prob",
        "factor_visibility_prob",
        "prototype_weights",
        "prior_scale",
    }
    required_observations = {
        "presence_target",
        "presence_mask",
        "visibility_target",
        "visibility_mask",
        "source_reliability",
        "geometry_mask",
        "geometry_mask_valid",
    }
    if not required_predictions <= set(predictions):
        raise KeyError(f"factor predictions missing {sorted(required_predictions - set(predictions))}")
    if not required_observations <= set(observations):
        raise KeyError(f"factor observations missing {sorted(required_observations - set(observations))}")

    presence_logits = predictions["factor_presence_logits"]
    visibility_logits = predictions["factor_visibility_logits"]
    source_reliability = observations["source_reliability"]
    presence_values = F.binary_cross_entropy_with_logits(
        presence_logits, observations["presence_target"].to(dtype=presence_logits.dtype), reduction="none"
    )
    visibility_values = F.binary_cross_entropy_with_logits(
        visibility_logits,
        observations["visibility_target"].to(dtype=visibility_logits.dtype),
        reduction="none",
    )
    loss_presence, count_presence = _factor_balanced_binary_mean(
        presence_values,
        observations["presence_target"].to(dtype=presence_logits.dtype),
        observations["presence_mask"],
        source_reliability,
    )
    loss_visibility, count_visibility = _masked_mean(
        visibility_values, observations["visibility_mask"], reliability=source_reliability
    )

    soft_masks = predictions["factor_soft_masks"]
    geometry_target = observations["geometry_mask"].to(dtype=soft_masks.dtype)
    if geometry_target.shape != soft_masks.shape:
        raise ValueError("factor geometry target must match factor soft masks")
    geometry_factor_valid = observations["geometry_mask_valid"] > 0
    geometry_valid = geometry_factor_valid[..., None, None].expand_as(soft_masks)
    geometry_reliability = source_reliability[..., None, None].expand_as(soft_masks)
    geometry_probability_fp32 = soft_masks.float().clamp(1e-6, 1.0 - 1e-6)
    geometry_values = F.binary_cross_entropy_with_logits(
        torch.logit(geometry_probability_fp32), geometry_target.float(), reduction="none"
    )
    loss_geometry, _ = _masked_mean(
        geometry_values, geometry_valid, reliability=geometry_reliability
    )
    count_geometry = geometry_factor_valid.sum().detach()

    view_presence_loss, count_view_presence = _optional_consistency(
        predictions["factor_presence_prob"],
        predictions.get("view_factor_presence_prob"),
        predictions.get("view_consistency_valid"),
    )
    view_visibility_loss, count_view_visibility = _optional_consistency(
        predictions["factor_visibility_prob"],
        predictions.get("view_factor_visibility_prob"),
        predictions.get("view_consistency_valid"),
    )
    view_count_sum = count_view_presence + count_view_visibility
    view_loss = torch.where(
        view_count_sum > 0,
        (
            view_presence_loss * count_view_presence.to(view_presence_loss.dtype)
            + view_visibility_loss * count_view_visibility.to(view_visibility_loss.dtype)
        )
        / view_count_sum.clamp_min(1).to(view_presence_loss.dtype),
        _zero(view_presence_loss),
    )
    count_view = view_count_sum
    flip_loss, count_flip = _optional_consistency(
        soft_masks,
        predictions.get("flip_factor_soft_masks_aligned"),
        predictions.get("flip_equivariance_valid"),
    )

    prototype_weights = predictions["prototype_weights"]
    if prototype_weights.ndim != 3:
        raise ValueError("prototype weights must be [B,F,K]")
    prototype_valid = predictions.get("prototype_valid_mask")
    if prototype_valid is None:
        prototype_valid = torch.ones(
            prototype_weights.shape[1:], device=prototype_weights.device, dtype=torch.bool
        )
    if tuple(prototype_valid.shape) != tuple(prototype_weights.shape[1:]):
        raise ValueError("prototype valid mask must be [F,K]")
    valid_float = prototype_valid.to(dtype=prototype_weights.dtype)
    occupancy = prototype_weights.mean(dim=0)
    uniform = valid_float / valid_float.sum(-1, keepdim=True).clamp_min(1.0)
    occupancy_values = (occupancy - uniform).square()
    loss_occupancy, count_occupancy = _masked_mean(occupancy_values, prototype_valid)

    pairwise_cosine = predictions.get("prototype_pairwise_cosine")
    if pairwise_cosine is None:
        loss_repulsion = _zero(prototype_weights)
        count_repulsion = prototype_weights.new_zeros((), dtype=torch.long)
    else:
        if pairwise_cosine.ndim != 3 or pairwise_cosine.shape[0] != prototype_valid.shape[0]:
            raise ValueError("prototype pairwise cosine must be [F,K,K]")
        pair_valid = prototype_valid[:, :, None] & prototype_valid[:, None, :]
        diagonal = torch.eye(
            pair_valid.shape[-1], device=pair_valid.device, dtype=torch.bool
        ).unsqueeze(0)
        pair_valid = pair_valid & ~diagonal
        loss_repulsion, count_repulsion = _masked_mean(
            pairwise_cosine.clamp_min(0.0).square(), pair_valid
        )

    prior_scale = predictions["prior_scale"]
    loss_prior_scale = prior_scale.square().mean() if prior_scale.numel() else _zero(prior_scale)
    count_prior_scale = prior_scale.new_tensor(prior_scale.numel(), dtype=torch.long)
    positive_evidence = predictions.get("factor_positive_evidence")
    contradiction_mask = predictions.get("factor_contradiction_mask")
    if positive_evidence is None or contradiction_mask is None:
        loss_contradiction = _zero(presence_logits)
        count_contradiction = presence_logits.new_zeros((), dtype=torch.long)
    else:
        factor_count = positive_evidence.shape[1]
        if tuple(contradiction_mask.shape) != (factor_count, factor_count):
            raise ValueError("factor contradiction mask must be [F,F]")
        if not torch.equal(contradiction_mask, contradiction_mask.T) or contradiction_mask.diagonal().any():
            raise ValueError("factor contradiction mask must be symmetric and irreflexive")
        upper = torch.triu(torch.ones_like(contradiction_mask, dtype=torch.bool), diagonal=1)
        pair_mask = contradiction_mask.to(device=positive_evidence.device, dtype=torch.bool) & upper
        contradiction_values = positive_evidence[:, :, None] * positive_evidence[:, None, :]
        contradiction_valid = pair_mask.unsqueeze(0).expand_as(contradiction_values)
        loss_contradiction, count_contradiction = _masked_mean(
            contradiction_values, contradiction_valid
        )

    total = (
        loss_presence
        + loss_visibility
        + geometry_mask_weight * loss_geometry
        + view_consistency_weight * view_loss
        + flip_equivariance_weight * flip_loss
        + prototype_occupancy_weight * loss_occupancy
        + prototype_repulsion_weight * loss_repulsion
        + prior_scale_weight * loss_prior_scale
        + contradiction_weight * loss_contradiction
    )
    return {
        "loss_presence": loss_presence,
        "loss_visibility": loss_visibility,
        "loss_geometry_mask": loss_geometry,
        "loss_view_consistency": view_loss,
        "loss_flip_equivariance": flip_loss,
        "loss_prototype_occupancy": loss_occupancy,
        "loss_prototype_repulsion": loss_repulsion,
        "loss_prior_scale": loss_prior_scale,
        "loss_contradiction": loss_contradiction,
        "loss_factor_total": total,
        "count_presence": count_presence,
        "count_visibility": count_visibility,
        "count_geometry_mask": count_geometry,
        "count_view_consistency": count_view,
        "count_flip_equivariance": count_flip,
        "count_prototype_occupancy": count_occupancy,
        "count_prototype_repulsion": count_repulsion,
        "count_prior_scale": count_prior_scale,
        "count_contradiction": count_contradiction,
    }
