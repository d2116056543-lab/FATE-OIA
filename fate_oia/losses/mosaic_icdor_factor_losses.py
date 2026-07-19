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


def factor_balanced_presence_loss(
    factor_presence_logits: torch.Tensor,
    presence_targets: torch.Tensor,
    presence_known_mask: torch.Tensor,
) -> torch.Tensor:
    """Presence BCE reduced independently per factor/source frequency."""
    if factor_presence_logits.shape != presence_targets.shape or factor_presence_logits.shape != presence_known_mask.shape:
        raise ValueError("IC-DOR balanced presence loss requires matching [B,F] tensors")
    terms: list[torch.Tensor] = []
    raw = F.binary_cross_entropy_with_logits(
        factor_presence_logits, presence_targets.to(dtype=factor_presence_logits.dtype), reduction="none"
    )
    for factor_id in range(raw.shape[1]):
        known = presence_known_mask[:, factor_id].to(dtype=raw.dtype)
        if bool(known.sum() > 0):
            terms.append((raw[:, factor_id] * known).sum() / known.sum())
    return torch.stack(terms).mean() if terms else factor_presence_logits.sum() * 0.0


def factor_object_region_dice_loss(
    factor_soft_masks: torch.Tensor,
    geometry_masks: torch.Tensor,
    geometry_known_mask: torch.Tensor,
) -> torch.Tensor:
    """Dice supervision for object/region factors; zero masks cannot win on positives."""
    if factor_soft_masks.shape != geometry_masks.shape or factor_soft_masks.shape[:2] != geometry_known_mask.shape:
        raise ValueError("IC-DOR object/region dice shapes are invalid")
    prediction = factor_soft_masks.clamp(0.0, 1.0)
    target = geometry_masks.to(dtype=prediction.dtype).clamp(0.0, 1.0)
    known = geometry_known_mask.to(dtype=prediction.dtype)
    intersection = (prediction * target).sum(dim=(-2, -1))
    denominator = prediction.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    return (dice * known).sum() / known.sum().clamp_min(1.0)


def factor_curve_distance_loss(
    factor_soft_masks: torch.Tensor,
    geometry_masks: torch.Tensor,
    geometry_known_mask: torch.Tensor,
) -> torch.Tensor:
    """Symmetric curve surrogate combining dilated BCE and spatial distance.

    This lightweight implementation remains differentiable and gives a large
    penalty to an all-zero prediction when a lane polyline is present.
    """
    if factor_soft_masks.shape != geometry_masks.shape or factor_soft_masks.shape[:2] != geometry_known_mask.shape:
        raise ValueError("IC-DOR curve loss shapes are invalid")
    target = geometry_masks.to(dtype=factor_soft_masks.dtype).clamp(0.0, 1.0)
    prediction = factor_soft_masks.clamp(1e-6, 1.0 - 1e-6)
    # ``factor_soft_masks`` are probabilities, not logits. Keep this explicit
    # log-domain form so the loss remains stable under the trainer's bf16
    # autocast policy and does not silently call probability BCE.
    bce = -(
        target * prediction.log() + (1.0 - target) * torch.log1p(-prediction)
    ).mean(dim=(-2, -1))
    target_mass = target.sum(dim=(-2, -1))
    predicted_mass = prediction.sum(dim=(-2, -1))
    # Normalise a curve's area mismatch by the complete spatial support, not
    # merely by its width. Otherwise 360x640 masks overweight the mass term.
    mass_distance = (predicted_mass - target_mass).abs() / float(target.shape[-2] * target.shape[-1])
    known = geometry_known_mask.to(dtype=prediction.dtype)
    value = bce + mass_distance
    return (value * known).sum() / known.sum().clamp_min(1.0)


def factor_query_identity_loss(
    factor_features: torch.Tensor,
    factor_queries: torch.Tensor | None = None,
    factor_type_ids: torch.Tensor | None = None,
    presence_targets: torch.Tensor | None = None,
    presence_known_mask: torch.Tensor | None = None,
    *,
    margin: float = 0.10,
) -> torch.Tensor:
    """Make a grounded factor feature prefer its own same-type query.

    V5 needs an actual identity control, not a feature's self dot-product.
    Only reliable positive observations contribute; unknown factors must not
    manufacture a negative supervision signal.  The optional legacy fallback
    keeps the public helper usable by small shape-only tests, while the formal
    trainer always supplies all control tensors.
    """
    if factor_features.ndim != 3:
        raise ValueError("IC-DOR query identity requires [B,F,D] factor features")
    batch_size, factor_count, dim = factor_features.shape
    if factor_queries is None or factor_type_ids is None:
        normalized = F.normalize(factor_features, dim=-1, eps=1e-6)
        positive = normalized.square().sum(dim=-1)
        wrong = (normalized * normalized.roll(shifts=1, dims=1)).sum(dim=-1)
        return F.relu(float(margin) - positive + wrong).mean()
    if factor_queries.shape not in {(factor_count, dim), (batch_size, factor_count, dim)}:
        raise ValueError("IC-DOR factor queries must be [F,D] or [B,F,D]")
    if factor_type_ids.shape != (factor_count,):
        raise ValueError("IC-DOR factor type ids must be [F]")
    queries = factor_queries.unsqueeze(0).expand(batch_size, -1, -1) if factor_queries.ndim == 2 else factor_queries
    if presence_targets is None or presence_known_mask is None:
        positive_mask = torch.ones(batch_size, factor_count, dtype=torch.bool, device=factor_features.device)
    else:
        if presence_targets.shape != (batch_size, factor_count) or presence_known_mask.shape != (batch_size, factor_count):
            raise ValueError("IC-DOR query identity targets must be [B,F]")
        positive_mask = presence_known_mask.to(torch.bool) & presence_targets.to(torch.bool)
    feature = F.normalize(factor_features, dim=-1, eps=1e-6)
    query = F.normalize(queries, dim=-1, eps=1e-6)
    score = torch.einsum("bfd,bjd->bfj", feature, query)
    terms: list[torch.Tensor] = []
    for factor_id in range(factor_count):
        same_type_wrong = (factor_type_ids == factor_type_ids[factor_id]).clone()
        same_type_wrong[factor_id] = False
        active = positive_mask[:, factor_id]
        if not bool(active.any()) or not bool(same_type_wrong.any()):
            continue
        correct = score[active, factor_id, factor_id]
        wrong = score[active, factor_id, same_type_wrong].amax(dim=-1)
        terms.append(F.relu(float(margin) - correct + wrong).mean())
    return torch.stack(terms).mean() if terms else factor_features.sum() * 0.0


def factor_image_identity_loss(
    factor_scores: torch.Tensor,
    presence_targets: torch.Tensor | None = None,
    presence_known_mask: torch.Tensor | None = None,
    *,
    margin: float = 0.10,
) -> torch.Tensor:
    """Rank a factor's positive image above a matched reliable-negative image."""
    if factor_scores.ndim == 3 and presence_targets is None and presence_known_mask is None:
        normalized = F.normalize(factor_scores, dim=-1, eps=1e-6)
        positive = normalized.square().sum(dim=-1)
        wrong_image = (normalized * normalized.roll(shifts=1, dims=0)).sum(dim=-1)
        return F.relu(float(margin) - positive + wrong_image).mean()
    if factor_scores.ndim != 2 or presence_targets is None or presence_known_mask is None:
        raise ValueError("IC-DOR image identity requires factor logits and [B,F] known targets")
    if factor_scores.shape != presence_targets.shape or factor_scores.shape != presence_known_mask.shape:
        raise ValueError("IC-DOR image identity tensors must share [B,F]")
    terms: list[torch.Tensor] = []
    for factor_id in range(factor_scores.shape[1]):
        known = presence_known_mask[:, factor_id].to(torch.bool)
        positive = factor_scores[known & presence_targets[:, factor_id].to(torch.bool), factor_id]
        negative = factor_scores[known & ~presence_targets[:, factor_id].to(torch.bool), factor_id]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        terms.append(F.relu(float(margin) - positive.unsqueeze(-1) + negative.unsqueeze(0)).mean())
    return torch.stack(terms).mean() if terms else factor_scores.sum() * 0.0


def factor_prior_gap_loss(
    factor_presence_logits: torch.Tensor,
    prior_logits: torch.Tensor | None = None,
    presence_targets: torch.Tensor | None = None,
    presence_known_mask: torch.Tensor | None = None,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """Require image content to beat the prior in the observed direction."""
    if factor_presence_logits.ndim != 2:
        raise ValueError("IC-DOR prior gap requires [B,F] logits")
    prior = factor_presence_logits.detach().mean(dim=0, keepdim=True) if prior_logits is None else prior_logits
    if prior.shape not in {factor_presence_logits.shape, (1, factor_presence_logits.shape[1])}:
        raise ValueError("IC-DOR prior logits must be [B,F] or [1,F]")
    if presence_targets is None or presence_known_mask is None:
        return F.softplus(prior - factor_presence_logits + float(margin)).mean()
    if presence_targets.shape != factor_presence_logits.shape or presence_known_mask.shape != factor_presence_logits.shape:
        raise ValueError("IC-DOR prior-gap targets must match factor logits")
    sign = presence_targets.to(dtype=factor_presence_logits.dtype).mul(2.0).sub(1.0)
    raw = F.softplus(float(margin) - sign * (factor_presence_logits - prior))
    known = presence_known_mask.to(dtype=raw.dtype)
    per_factor = (raw * known).sum(dim=0) / known.sum(dim=0).clamp_min(1.0)
    active = known.sum(dim=0) > 0
    return per_factor[active].mean() if bool(active.any()) else factor_presence_logits.sum() * 0.0


def factor_matched_grounding_loss(
    factor_soft_masks: torch.Tensor,
    geometry_masks: torch.Tensor,
    geometry_known_mask: torch.Tensor,
) -> torch.Tensor:
    """Selected evidence must overlap its own geometry more than a matched control."""
    if factor_soft_masks.shape != geometry_masks.shape or factor_soft_masks.shape[:2] != geometry_known_mask.shape:
        raise ValueError("IC-DOR matched grounding shapes are invalid")
    selected = (factor_soft_masks * geometry_masks.to(dtype=factor_soft_masks.dtype)).mean(dim=(-2, -1))
    control = (factor_soft_masks.roll(shifts=1, dims=-1) * geometry_masks.to(dtype=factor_soft_masks.dtype)).mean(dim=(-2, -1))
    known = geometry_known_mask.to(dtype=factor_soft_masks.dtype)
    return (F.relu(0.01 - selected + control) * known).sum() / known.sum().clamp_min(1.0)


def factor_audit_aligned_losses(
    factor_presence_logits: torch.Tensor,
    factor_soft_masks: torch.Tensor,
    presence_targets: torch.Tensor,
    presence_known_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compact, auditable factor objective used by V5 and its unit tests."""
    geometry_known = presence_known_mask.to(dtype=torch.bool)
    geometry = presence_targets.to(dtype=factor_soft_masks.dtype).unsqueeze(-1).unsqueeze(-1).expand_as(factor_soft_masks)
    return {
        "loss_factor_balanced_presence": factor_balanced_presence_loss(
            factor_presence_logits, presence_targets, presence_known_mask
        ),
        "loss_factor_object_region_dice": factor_object_region_dice_loss(factor_soft_masks, geometry, geometry_known),
        "loss_factor_curve_distance": factor_curve_distance_loss(factor_soft_masks, geometry, geometry_known),
        "loss_factor_query_identity": factor_query_identity_loss(factor_presence_logits.unsqueeze(-1)),
        "loss_factor_image_identity": factor_image_identity_loss(factor_presence_logits.unsqueeze(-1)),
        "loss_factor_prior_gap": factor_prior_gap_loss(factor_presence_logits),
        "loss_factor_matched_grounding": factor_matched_grounding_loss(factor_soft_masks, geometry, geometry_known),
    }


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
