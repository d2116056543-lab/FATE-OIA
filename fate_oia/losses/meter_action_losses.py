from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def asymmetric_multilabel_elements(
    logits: Tensor, target: Tensor, gamma_negative: float = 2.0
) -> Tensor:
    probability = torch.sigmoid(logits)
    positive = -target * torch.log(probability.clamp_min(1e-6))
    negative = -(1.0 - target) * probability.pow(gamma_negative) * torch.log((1.0 - probability).clamp_min(1e-6))
    return positive + negative


def asymmetric_multilabel_loss(logits: Tensor, target: Tensor, gamma_negative: float = 2.0) -> Tensor:
    return asymmetric_multilabel_elements(logits, target, gamma_negative).mean()


def soft_f1_loss(logits: Tensor, target: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum(dim=0)
    denominator = probability.sum(dim=0) + target.sum(dim=0) + 1e-6
    return 1.0 - (numerator / denominator).mean()


def meter_action_loss_per_sample(
    output: dict[str, Tensor],
    target: Tensor,
    weights: dict[str, float] | None = None,
) -> Tensor:
    """Return the formal action surrogate at sample grain for meta audit."""
    weights = weights or {}

    def asymmetric_per_sample(logits: Tensor) -> Tensor:
        probability = torch.sigmoid(logits)
        positive = -target * torch.log(probability.clamp_min(1e-6))
        negative = (
            -(1.0 - target)
            * probability.pow(2.0)
            * torch.log((1.0 - probability).clamp_min(1e-6))
        )
        return (positive + negative).mean(dim=-1)

    final = asymmetric_per_sample(output["action_logits_final"])
    visual = asymmetric_per_sample(output["action_logits_visual"])
    semantic = asymmetric_per_sample(output["action_logits_semantic"])
    two_way = F.binary_cross_entropy_with_logits(
        output["action_logits_peer"], target, reduction="none"
    ).mean(dim=-1)
    probability = torch.sigmoid(output["action_logits_final"])
    # The formal soft-F1 couples samples through per-label batch statistics.
    # Broadcasting the exact scalar preserves both the mean objective and its
    # gradient while still exposing a per-sample vector for paired auditing.
    soft_f1 = soft_f1_loss(
        output["action_logits_final"], target
    ).expand(target.shape[0])
    cardinality = F.smooth_l1_loss(
        probability.sum(dim=-1), target.sum(dim=-1), reduction="none"
    )
    peer = asymmetric_multilabel_elements(
        output.get("action_logits_peer_regret", output["action_logits_peer"]),
        target,
    )
    visual_per = asymmetric_multilabel_elements(
        output["action_logits_visual"].detach(), target
    )
    semantic_per = asymmetric_multilabel_elements(
        output.get(
            "action_logits_semantic_transport",
            output["action_logits_semantic"],
        ).detach(),
        target,
    )
    selector_regret = torch.relu(
        peer - torch.minimum(visual_per, semantic_per) + 1e-4
    ).mean(dim=-1)
    return (
        weights.get("action_final", 1.0) * final
        + weights.get("action_visual", 0.40) * visual
        + weights.get("action_semantic", 0.40) * semantic
        + weights.get("action_two_way", 0.05) * two_way
        + weights.get("action_soft_f1", 0.03) * soft_f1
        + weights.get("action_cardinality", 0.02) * cardinality
        + weights.get("selector_regret", 0.10) * selector_regret
    )


def meter_action_loss(output: dict[str, Tensor], target: Tensor, weights: dict[str, float] | None = None) -> dict[str, Tensor]:
    weights = weights or {}
    final = asymmetric_multilabel_loss(output["action_logits_final"], target)
    visual = asymmetric_multilabel_loss(output["action_logits_visual"], target)
    semantic = asymmetric_multilabel_loss(output["action_logits_semantic"], target)
    two_way = F.binary_cross_entropy_with_logits(output["action_logits_peer"], target)
    soft_f1 = soft_f1_loss(output["action_logits_final"], target)
    cardinality = F.smooth_l1_loss(torch.sigmoid(output["action_logits_final"]).sum(dim=-1), target.sum(dim=-1))
    peer = asymmetric_multilabel_elements(
        output.get("action_logits_peer_regret", output["action_logits_peer"]),
        target,
    )
    visual_per = asymmetric_multilabel_elements(
        output["action_logits_visual"].detach(), target
    )
    semantic_per = asymmetric_multilabel_elements(
        output.get(
            "action_logits_semantic_transport",
            output["action_logits_semantic"],
        ).detach(),
        target,
    )
    selector_regret = torch.relu(peer - torch.minimum(visual_per, semantic_per) + 1e-4).mean()
    total = (
        weights.get("action_final", 1.0) * final
        + weights.get("action_visual", 0.40) * visual
        + weights.get("action_semantic", 0.40) * semantic
        + weights.get("action_two_way", 0.05) * two_way
        + weights.get("action_soft_f1", 0.03) * soft_f1
        + weights.get("action_cardinality", 0.02) * cardinality
        + weights.get("selector_regret", 0.10) * selector_regret
    )
    return {
        "final": final,
        "visual": visual,
        "semantic": semantic,
        "two_way": two_way,
        "soft_f1": soft_f1,
        "cardinality": cardinality,
        "selector_regret": selector_regret,
        "total": total,
    }
