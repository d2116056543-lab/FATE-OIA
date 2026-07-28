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
    evidence = evidence.detach().clamp(0.0, 1.0)
    observability = (
        torch.ones_like(evidence)
        if observability is None
        else observability.detach().clamp(0.0, 1.0)
    )
    positive_weight = 0.5 + 0.5 * evidence
    negative_weight = 0.1 + 0.3 * observability * (1.0 - evidence)
    positive = -positive_weight * target * torch.log(probability.clamp_min(1e-6))
    negative = -negative_weight * (1.0 - target) * probability.square() * torch.log((1.0 - probability).clamp_min(1e-6))
    return (positive + negative).mean()


def reason_soft_f1(
    logits: Tensor,
    target: Tensor,
    evidence: Tensor,
    observability: Tensor | None = None,
) -> Tensor:
    probability = torch.sigmoid(logits)
    evidence = evidence.detach().clamp(0.0, 1.0)
    observability = (
        torch.ones_like(evidence)
        if observability is None
        else observability.detach().clamp(0.0, 1.0)
    )
    weight = target + (1.0 - target) * (0.1 + 0.3 * observability * (1.0 - evidence))
    numerator = 2.0 * (probability * target * weight).sum(dim=0)
    denominator = (probability * weight).sum(dim=0) + (target * weight).sum(dim=0) + 1e-6
    return 1.0 - (numerator / denominator).mean()


def tail_rank_loss(logits: Tensor, target: Tensor) -> Tensor:
    positive = logits.masked_fill(target < 0.5, float("-inf")).amax(dim=-1)
    negative = logits.masked_fill(target > 0.5, float("inf")).amin(dim=-1)
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
    # The private candidate is supervised from update zero even while the
    # public final branch is still interpolating from CalAlign.
    candidate_logits = output.get("reason_logits_candidate", output["reason_logits_final"])
    final = weighted_reason_asl(candidate_logits, target, confidence, observability)
    global_view = weighted_reason_asl(output["reason_logits_global"], target, confidence, observability)
    local = weighted_reason_asl(output["reason_logits_local"], target, confidence, observability)
    rank = tail_rank_loss(output["reason_logits_final"], target)
    soft_f1 = reason_soft_f1(candidate_logits, target, confidence, observability)
    annotation_delta = output["reason_annotation_delta"].square().mean()
    total = (
        weights.get("reason_final", 1.0) * final
        + weights.get("reason_global", 0.40) * global_view
        + weights.get("reason_local", 0.40) * local
        + weights.get("reason_rank", 0.05) * rank
        + weights.get("reason_soft_f1", 0.05) * soft_f1
        + weights.get("reason_annotation_delta", 0.02) * annotation_delta
    )
    return {
        "final": final,
        "global": global_view,
        "local": local,
        "rank": rank,
        "soft_f1": soft_f1,
        "annotation_delta": annotation_delta,
        "total": total,
    }
