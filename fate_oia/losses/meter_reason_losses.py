from __future__ import annotations

import torch
from torch import Tensor


def weighted_reason_asl(
    logits: Tensor,
    target: Tensor,
    evidence: Tensor,
    observability: Tensor | None = None,
) -> Tensor:
    probability = torch.sigmoid(logits)
    evidence = evidence.detach().clamp(0, 1)
    obs = torch.ones_like(evidence) if observability is None else observability.detach().clamp(0, 1)
    positive = -(0.5 + 0.5 * evidence) * target * torch.log(
        probability.clamp_min(1e-6)
    )
    negative_weight = 0.1 + 0.3 * obs * (1.0 - evidence)
    negative = (
        -negative_weight
        * (1.0 - target)
        * probability.square()
        * torch.log((1.0 - probability).clamp_min(1e-6))
    )
    return (positive + negative).mean()


def reason_soft_f1(
    logits: Tensor, target: Tensor, evidence: Tensor, observability: Tensor | None = None
) -> Tensor:
    probability = torch.sigmoid(logits)
    obs = torch.ones_like(evidence) if observability is None else observability.detach().clamp(0, 1)
    negative_weight = 0.1 + 0.3 * obs * (1.0 - evidence.detach().clamp(0, 1))
    weight = target + (1.0 - target) * negative_weight
    numerator = 2 * (probability * target * weight).sum(0)
    denominator = (probability * weight).sum(0) + (target * weight).sum(0) + 1e-6
    return 1.0 - (numerator / denominator).mean()


def tail_rank_loss(
    logits: Tensor, target: Tensor, negative_weight: Tensor | None = None
) -> Tensor:
    if negative_weight is None:
        negative_weight = torch.ones_like(target)
    reliable_negative = (1.0 - target) * negative_weight.detach().clamp(0, 1)
    pair_weight = target.unsqueeze(-1) * reliable_negative.unsqueeze(-2)
    if not bool(pair_weight.gt(0).any()):
        return logits.new_zeros(())
    margin = torch.relu(
        0.20 - logits.unsqueeze(-1) + logits.unsqueeze(-2)
    )
    return (margin * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)


def meter_reason_loss(
    output: dict[str, Tensor],
    target: Tensor,
    confidence: Tensor,
    weights: dict[str, float] | None = None,
    *,
    observability: Tensor | None = None,
) -> dict[str, Tensor]:
    weights = weights or {}
    final = weighted_reason_asl(
        output["reason_logits_final"], target, confidence, observability
    )
    global_view = weighted_reason_asl(
        output["reason_logits_global"], target, confidence, observability
    )
    obs = torch.ones_like(confidence) if observability is None else observability.detach().clamp(0, 1)
    reliable_negative = obs * (1.0 - confidence.detach().clamp(0, 1))
    rank = tail_rank_loss(
        output["reason_logits_final"], target, negative_weight=reliable_negative
    )
    soft_f1 = reason_soft_f1(
        output["reason_logits_final"], target, confidence, observability
    )
    global_logits = output["reason_logits_global"].detach()
    final_logits = output["reason_logits_final"]
    positive_guard = target * torch.relu(global_logits - final_logits)
    negative_guard = (1.0 - target) * reliable_negative * torch.relu(final_logits - global_logits)
    correction_weight = target + (1.0 - target) * reliable_negative
    correction = (positive_guard + negative_guard).sum() / correction_weight.sum().clamp_min(1e-6)
    total = (
        weights.get("reason_final", 1.0) * final
        + weights.get("reason_global", 0.45) * global_view
        + weights.get("reason_rank", 0.05) * rank
        + weights.get("reason_soft_f1", 0.05) * soft_f1
        + weights.get("reason_evidence_correction", 0.03) * correction
    )
    return {
        "final": final,
        "global": global_view,
        "rank": rank,
        "soft_f1": soft_f1,
        "evidence_correction": correction,
        "total": total,
    }


def meter_reason_share_loss(*args: object, **kwargs: object) -> dict[str, Tensor]:
    raise RuntimeError("Formal meta-sharing is disabled in TESA")
