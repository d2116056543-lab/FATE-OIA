from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def asymmetric_multilabel_elements(
    logits: Tensor, target: Tensor, gamma_negative: float = 2.0
) -> Tensor:
    probability = torch.sigmoid(logits)
    positive = -target * torch.log(probability.clamp_min(1e-6))
    negative = (
        -(1.0 - target)
        * probability.pow(gamma_negative)
        * torch.log((1.0 - probability).clamp_min(1e-6))
    )
    return positive + negative


def asymmetric_multilabel_loss(logits: Tensor, target: Tensor) -> Tensor:
    return asymmetric_multilabel_elements(logits, target).mean()


def soft_f1_loss(logits: Tensor, target: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum(0)
    denominator = probability.sum(0) + target.sum(0) + 1e-6
    return 1.0 - (numerator / denominator).mean()


def action_transport_anti_monopoly_loss(
    factor_weights: Tensor,
    source_mask: Tensor,
    *,
    max_share: float = 0.70,
) -> Tensor:
    """Penalize only avoidable factor monopoly, never sparse valid evidence."""
    available = source_mask.gt(1e-8)
    available_count = available.sum(-1)
    masked_weights = factor_weights * available.to(factor_weights.dtype)
    normalized = masked_weights / masked_weights.sum(-1, keepdim=True).clamp_min(1e-8)
    dominance = torch.relu(normalized.max(-1).values - float(max_share))
    valid = available_count > 1
    if not bool(valid.any()):
        return factor_weights.new_zeros(())
    return dominance[valid].mean()


def action_delta_pairwise_ranking_loss(
    action_evidence_delta: Tensor,
    target: Tensor,
    *,
    normalizer: Tensor | None = None,
    margin: float = 0.10,
    temperature: float = 0.25,
) -> Tensor:
    """Smooth, bounded ranking for the evidence correction only.

    The normalized correction retains positive-over-negative ordering without
    allowing an unstable auxiliary branch to dominate the action objective.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if normalizer is None:
        normalizer = action_evidence_delta.detach().abs().amax(0).clamp_min(1.0)
    normalizer = normalizer.detach().to(action_evidence_delta)
    if normalizer.ndim == 1:
        normalizer = normalizer.unsqueeze(0)
    normalized_delta = torch.tanh(
        action_evidence_delta / normalizer.clamp_min(1e-6)
    )
    terms: list[Tensor] = []
    for action_id in range(normalized_delta.shape[1]):
        positive = target[:, action_id] > 0.5
        negative = ~positive
        if not bool(positive.any()) or not bool(negative.any()):
            continue
        difference = (
            normalized_delta[positive, action_id].unsqueeze(1)
            - normalized_delta[negative, action_id].unsqueeze(0)
        )
        terms.append(
            (
                float(temperature)
                * F.softplus((float(margin) - difference) / float(temperature))
            ).mean()
        )
    if not terms:
        return action_evidence_delta.new_zeros(())
    return torch.stack(terms).mean()


def meter_action_loss(
    output: dict[str, Tensor],
    target: Tensor,
    weights: dict[str, float] | None = None,
) -> dict[str, Tensor]:
    weights = weights or {}
    final = asymmetric_multilabel_loss(output["action_logits_final"], target)
    visual = asymmetric_multilabel_loss(output["action_logits_visual"], target)
    sign = target * 2.0 - 1.0
    visual_margin = sign * output["action_logits_visual"].detach()
    final_margin = sign * output["action_logits_final"]
    support = output.get("action_transport_support", torch.ones_like(final_margin))
    admission = output.get("action_evidence_admission_gate", torch.ones_like(final_margin))
    correction_weight = (support * admission).detach().clamp_min(0.0)
    correction_terms = torch.relu(visual_margin - final_margin) * correction_weight
    correction = correction_terms.sum() / correction_weight.sum().clamp_min(1e-6)
    two_way = F.binary_cross_entropy_with_logits(
        output["action_logits_final"], target
    )
    soft_f1 = soft_f1_loss(output["action_logits_final"], target)
    cardinality = F.smooth_l1_loss(
        torch.sigmoid(output["action_logits_final"]).sum(-1), target.sum(-1)
    )
    identity = output.get(
        "dense_identity_loss", output["action_logits_final"].new_zeros(())
    )
    # The explicit key avoids re-consuming the legacy dense intervention term,
    # which is already added by the trainer's dense loss total.
    specificity = output.get(
        "action_specificity_loss", output["action_logits_final"].new_zeros(())
    )
    anti_monopoly = output.get("action_anti_monopoly_loss")
    if anti_monopoly is None:
        anti_monopoly = action_transport_anti_monopoly_loss(
            output.get(
                "action_factor_dense_weights",
                output.get(
                    "action_factor_weights",
                    output["action_logits_final"].new_zeros(
                        output["action_logits_final"].shape[0],
                        output["action_logits_final"].shape[1],
                        1,
                    ),
                ),
            ),
            output.get(
                "action_factor_source_mask",
                output["action_logits_final"].new_zeros(
                    output["action_logits_final"].shape[0],
                    output["action_logits_final"].shape[1],
                    1,
                ),
            ),
        )
    near_boundary = output.get("action_near_boundary_loss")
    if near_boundary is None:
        from .meter_counterfactual_losses import near_boundary_delta_ranking_loss

        near_boundary = near_boundary_delta_ranking_loss(
            output["action_logits_visual"],
            output["action_logits_final"] - output["action_logits_visual"],
            target,
        )
    delta_ranking = action_delta_pairwise_ranking_loss(
        output.get(
            "action_evidence_delta",
            output["action_logits_final"] - output["action_logits_visual"],
        ),
        target,
        normalizer=output.get("action_correction_kappa"),
    )
    total = (
        weights.get("action_final", 1.00) * final
        + weights.get("action_visual", 0.35) * visual
        + weights.get("action_correction", 0.20) * correction
        + weights.get("action_two_way", 0.05) * two_way
        + weights.get("action_soft_f1", 0.03) * soft_f1
        + weights.get("action_cardinality", 0.02) * cardinality
        + weights.get("action_specificity", 0.0) * specificity
        + weights.get("action_identity", 0.03) * identity
        + weights.get("action_anti_monopoly", 0.0) * anti_monopoly
        + weights.get("action_near_boundary", 0.0) * near_boundary
        + weights.get("action_delta_ranking", 0.0) * delta_ranking
    )
    return {
        "final": final,
        "visual": visual,
        "correction": correction,
        "two_way": two_way,
        "soft_f1": soft_f1,
        "cardinality": cardinality,
        "specificity": specificity,
        "identity": identity,
        "anti_monopoly": anti_monopoly,
        "near_boundary": near_boundary,
        "delta_ranking": delta_ranking,
        "total": total,
    }


def meter_action_loss_per_sample(
    output: dict[str, Tensor],
    target: Tensor,
    weights: dict[str, float] | None = None,
) -> Tensor:
    weights = weights or {}
    final = asymmetric_multilabel_elements(
        output["action_logits_final"], target
    ).mean(-1)
    visual = asymmetric_multilabel_elements(
        output["action_logits_visual"], target
    ).mean(-1)
    correction_weight = (
        output.get("action_transport_support", torch.ones_like(target))
        * output.get("action_evidence_admission_gate", torch.ones_like(target))
    ).detach().clamp_min(0.0)
    correction_terms = torch.relu(
        (target * 2 - 1) * output["action_logits_visual"].detach()
        - (target * 2 - 1) * output["action_logits_final"]
    ) * correction_weight
    correction = correction_terms.sum(-1) / correction_weight.sum(-1).clamp_min(1e-6)
    return (
        weights.get("action_final", 1.0) * final
        + weights.get("action_visual", 0.35) * visual
        + weights.get("action_correction", 0.20) * correction
    )
