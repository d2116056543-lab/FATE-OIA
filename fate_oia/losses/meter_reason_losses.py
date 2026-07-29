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


def reason_soft_f1(logits: Tensor, target: Tensor, evidence: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    negative_weight = 0.1 + 0.3 * (1.0 - evidence.detach().clamp(0, 1))
    weight = target + (1.0 - target) * negative_weight
    numerator = 2 * (probability * target * weight).sum(0)
    denominator = (probability * weight).sum(0) + (target * weight).sum(0) + 1e-6
    return 1.0 - (numerator / denominator).mean()


def tail_rank_loss(logits: Tensor, target: Tensor) -> Tensor:
    positive = logits.masked_fill(target < 0.5, float("-inf")).amax(-1)
    negative = logits.masked_fill(target > 0.5, float("inf")).amin(-1)
    valid = torch.isfinite(positive) & torch.isfinite(negative)
    if not bool(valid.any()):
        return logits.new_zeros(())
    return torch.relu(0.20 - positive[valid] + negative[valid]).mean()


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
    rank = tail_rank_loss(output["reason_logits_final"], target)
    soft_f1 = reason_soft_f1(output["reason_logits_final"], target, confidence)
    sign = target * 2 - 1
    correction = torch.relu(
        0.02
        + sign * output["reason_logits_global"].detach()
        - sign * output["reason_logits_final"]
    ).mean()
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
