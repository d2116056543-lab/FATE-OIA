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
    route_strength_target: float = 0.05,
    shadow_asl_weight: float = 1.0,
    pareto_weight: float = 1.0,
    sparsity_weight: float = 0.02,
    dustbin_weight: float = 0.01,
    strength_weight: float = 0.02,
    intervention_weight: float = 0.05,
) -> dict[str, torch.Tensor]:
    if any(value.shape != action_visual_logits.shape for value in (action_support_logits, action_veto_logits, action_targets)):
        raise ValueError("IC-DOR route losses require matching [B,4] action tensors")
    route_logits = action_visual_logits.detach() + action_support_logits - action_veto_logits
    loss_asl = asymmetric_loss(route_logits, action_targets)
    loss_sparsity = (action_support_logits.abs() + action_veto_logits.abs()).mean()
    loss_dustbin = (1.0 - support_dustbin).mean() * (1.0 - veto_dustbin).mean()
    strength = (action_support_logits + action_veto_logits).mean()
    loss_strength = F.smooth_l1_loss(strength, route_logits.new_tensor(float(route_strength_target)))
    if matched_random_logits is None:
        loss_intervention = route_logits.sum() * 0.0
    else:
        if matched_random_logits.shape != route_logits.shape:
            raise ValueError("IC-DOR matched-random route logits must be [B,4]")
        selected_loss = F.binary_cross_entropy_with_logits(route_logits, action_targets.to(route_logits.dtype), reduction="none")
        random_loss = F.binary_cross_entropy_with_logits(matched_random_logits, action_targets.to(route_logits.dtype), reduction="none")
        loss_intervention = F.relu(0.01 + selected_loss - random_loss.detach()).mean()
    total = (
        shadow_asl_weight * loss_asl + pareto_weight * pareto_penalty
        + sparsity_weight * loss_sparsity + dustbin_weight * loss_dustbin
        + strength_weight * loss_strength + intervention_weight * loss_intervention
    )
    return {
        "action_shadow_logits": route_logits,
        "loss_action_route_asl": loss_asl,
        "loss_action_route_pareto": pareto_penalty,
        "loss_action_route_sparsity": loss_sparsity,
        "loss_action_route_dustbin": loss_dustbin,
        "loss_action_route_strength": loss_strength,
        "loss_action_route_intervention": loss_intervention,
        "loss_action_route_total": total,
    }
