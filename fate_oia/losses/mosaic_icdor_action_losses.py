from __future__ import annotations

import torch
from torch.nn import functional as F


def asymmetric_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
) -> torch.Tensor:
    """True multi-label ASL, not BCE under an ASL name."""
    if logits.shape != targets.shape or logits.ndim != 2:
        raise ValueError("IC-DOR ASL expects matching rank-2 logits and binary targets")
    if gamma_pos < 0.0 or gamma_neg < 0.0 or not 0.0 <= clip < 1.0:
        raise ValueError("IC-DOR ASL hyperparameters are invalid")
    targets = targets.to(dtype=logits.dtype)
    probability = torch.sigmoid(logits)
    positive = targets * (1.0 - probability).pow(gamma_pos) * F.logsigmoid(logits)
    negative_probability = (probability - clip).clamp_min(0.0)
    negative = (1.0 - targets) * negative_probability.pow(gamma_neg) * torch.log1p(-negative_probability.clamp(max=1.0 - 1e-6))
    return -(positive + negative).mean()


def _cross_sample_rank(logits: torch.Tensor, targets: torch.Tensor, *, margin: float = 0.10) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for label_index in range(logits.shape[1]):
        positives = logits[targets[:, label_index] > 0.5, label_index]
        negatives = logits[targets[:, label_index] <= 0.5, label_index]
        if positives.numel() and negatives.numel():
            losses.append(F.relu(margin - positives[:, None] + negatives[None, :]).mean())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def action_base_losses(
    action_visual_logits: torch.Tensor,
    action_targets: torch.Tensor,
    *,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    base_asl_weight: float = 1.0,
    rank_weight: float = 0.10,
    cardinality_weight: float = 0.02,
) -> dict[str, torch.Tensor]:
    if action_visual_logits.shape != action_targets.shape or action_visual_logits.shape[-1] != 4:
        raise ValueError("IC-DOR base action losses require [B,4]")
    loss_asl = asymmetric_loss(action_visual_logits, action_targets, gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip)
    loss_rank = _cross_sample_rank(action_visual_logits, action_targets)
    loss_cardinality = F.smooth_l1_loss(
        torch.sigmoid(action_visual_logits).sum(dim=-1),
        action_targets.to(dtype=action_visual_logits.dtype).sum(dim=-1),
    )
    total = base_asl_weight * loss_asl + rank_weight * loss_rank + cardinality_weight * loss_cardinality
    return {
        "loss_action_base_asl": loss_asl,
        "loss_action_base_rank": loss_rank,
        "loss_action_base_cardinality": loss_cardinality,
        "loss_action_base_total": total,
    }


def directional_credit_loss(
    selected_effect: torch.Tensor,
    control_effect: torch.Tensor,
    target_direction: torch.Tensor,
    *,
    margin: float = 0.01,
) -> torch.Tensor:
    """Selected evidence must beat its matched control in the target direction."""
    if selected_effect.shape != control_effect.shape or selected_effect.shape != target_direction.shape:
        raise ValueError("IC-DOR directional credit requires matching tensors")
    signed_effect = (selected_effect - control_effect.detach()) * target_direction.to(selected_effect.dtype)
    return F.relu(float(margin) - signed_effect).mean()


def wrong_target_specificity_loss(
    selected_effect: torch.Tensor,
    wrong_target_effect: torch.Tensor,
    *,
    margin: float = 0.05,
) -> torch.Tensor:
    """Correct target evidence must not be indistinguishable from a wrong target."""
    if selected_effect.shape != wrong_target_effect.shape:
        raise ValueError("IC-DOR wrong-target specificity requires matching tensors")
    return F.relu(float(margin) - (selected_effect - wrong_target_effect.detach()).abs()).mean()


def action_route_losses(
    action_visual_logits: torch.Tensor,
    action_support_logits: torch.Tensor,
    action_veto_logits: torch.Tensor,
    action_targets: torch.Tensor,
    *,
    support_dustbin: torch.Tensor,
    veto_dustbin: torch.Tensor,
    pareto_penalty: torch.Tensor,
    matched_random_logits: torch.Tensor | None = None,
    shadow_asl_weight: float = 1.0,
    pareto_weight: float = 1.0,
    sparsity_weight: float = 0.02,
    dustbin_weight: float = 0.01,
    intervention_weight: float = 0.05,
    selected_control_logits: torch.Tensor | None = None,
    wrong_target_logits: torch.Tensor | None = None,
    target_direction: torch.Tensor | None = None,
    independent_evidence_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if any(value.shape != action_visual_logits.shape for value in (action_support_logits, action_veto_logits, action_targets)):
        raise ValueError("IC-DOR route losses require matching [B,4] action tensors")
    route_logits = action_visual_logits.detach() + action_support_logits - action_veto_logits
    loss_asl = asymmetric_loss(route_logits, action_targets)
    loss_sparsity = (action_support_logits.abs() + action_veto_logits.abs()).mean()
    loss_dustbin = (1.0 - support_dustbin).mean() * (1.0 - veto_dustbin).mean()
    # V4 forced every route toward an arbitrary 0.05 correction. V5 only
    # constrains route magnitude when an independent evidence audit says the
    # edge is meaningful; otherwise zero mass is a valid and desirable state.
    if independent_evidence_mask is None:
        loss_band = route_logits.sum() * 0.0
    else:
        if independent_evidence_mask.shape != route_logits.shape:
            raise ValueError("IC-DOR route evidence mask must be [B,4]")
        magnitude = (action_support_logits - action_veto_logits).abs()
        band_low = F.relu(0.005 - magnitude)
        band_high = F.relu(magnitude - 0.20)
        mask = independent_evidence_mask.to(dtype=route_logits.dtype)
        loss_band = ((band_low + band_high) * mask).sum() / mask.sum().clamp_min(1.0)
    if matched_random_logits is None:
        loss_intervention = route_logits.sum() * 0.0
    else:
        if matched_random_logits.shape != route_logits.shape:
            raise ValueError("IC-DOR matched-random route logits must be [B,4]")
        selected_loss = F.binary_cross_entropy_with_logits(route_logits, action_targets.to(route_logits.dtype), reduction="none")
        random_loss = F.binary_cross_entropy_with_logits(matched_random_logits, action_targets.to(route_logits.dtype), reduction="none")
        loss_intervention = F.relu(0.01 + selected_loss - random_loss.detach()).mean()
    if selected_control_logits is None or target_direction is None:
        loss_credit = route_logits.sum() * 0.0
    else:
        if selected_control_logits.shape != route_logits.shape:
            raise ValueError("IC-DOR selected-control logits must be [B,4]")
        loss_credit = directional_credit_loss(
            route_logits - action_visual_logits.detach(),
            selected_control_logits - action_visual_logits.detach(),
            target_direction,
        )
    if wrong_target_logits is None:
        loss_specificity = route_logits.sum() * 0.0
    else:
        if wrong_target_logits.shape != route_logits.shape:
            raise ValueError("IC-DOR wrong-target logits must be [B,4]")
        loss_specificity = wrong_target_specificity_loss(
            route_logits - action_visual_logits.detach(),
            wrong_target_logits - action_visual_logits.detach(),
        )
    total = (
        shadow_asl_weight * loss_asl + pareto_weight * pareto_penalty
        + sparsity_weight * loss_sparsity + dustbin_weight * loss_dustbin
        + intervention_weight * loss_intervention + 0.10 * loss_credit
        + 0.05 * loss_specificity + 0.05 * loss_band
    )
    return {
        "action_shadow_logits": route_logits,
        "loss_action_route_asl": loss_asl,
        "loss_action_route_pareto": pareto_penalty,
        "loss_action_route_sparsity": loss_sparsity,
        "loss_action_route_dustbin": loss_dustbin,
        "loss_action_route_band": loss_band,
        "loss_action_route_intervention": loss_intervention,
        "loss_action_route_directional_credit": loss_credit,
        "loss_action_route_wrong_target_specificity": loss_specificity,
        "loss_action_route_total": total,
    }
