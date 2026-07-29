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
    correction = torch.relu(0.05 + visual_margin - final_margin).mean()
    two_way = F.binary_cross_entropy_with_logits(
        output["action_logits_final"], target
    )
    soft_f1 = soft_f1_loss(output["action_logits_final"], target)
    cardinality = F.smooth_l1_loss(
        torch.sigmoid(output["action_logits_final"]).sum(-1), target.sum(-1)
    )
    specificity = output.get(
        "dense_specificity_loss", output["action_logits_final"].new_zeros(())
    )
    identity = output.get(
        "dense_identity_loss", output["action_logits_final"].new_zeros(())
    )
    total = (
        weights.get("action_final", 1.00) * final
        + weights.get("action_visual", 0.35) * visual
        + weights.get("action_correction", 0.20) * correction
        + weights.get("action_two_way", 0.05) * two_way
        + weights.get("action_soft_f1", 0.03) * soft_f1
        + weights.get("action_cardinality", 0.02) * cardinality
        + weights.get("action_identity", 0.03) * identity
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
    correction = torch.relu(
        0.05
        + (target * 2 - 1) * output["action_logits_visual"].detach()
        - (target * 2 - 1) * output["action_logits_final"]
    ).mean(-1)
    return (
        weights.get("action_final", 1.0) * final
        + weights.get("action_visual", 0.35) * visual
        + weights.get("action_correction", 0.20) * correction
    )
