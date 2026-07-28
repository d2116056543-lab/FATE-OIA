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


def weighted_reason_asl_elements(
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
    negative = (
        -negative_weight
        * (1.0 - target)
        * probability.square()
        * torch.log((1.0 - probability).clamp_min(1e-6))
    )
    return positive + negative


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
    if "reason_logits_mix" in output:
        mix_peer = weighted_reason_asl_elements(
            output.get("reason_logits_mix_regret", output["reason_logits_mix"]),
            target,
            confidence,
            observability,
        )
        global_per = weighted_reason_asl_elements(
            output["reason_logits_global"].detach(),
            target,
            confidence,
            observability,
        )
        local_per = weighted_reason_asl_elements(
            output["reason_logits_local"].detach(),
            target,
            confidence,
            observability,
        )
        mix_regret = torch.relu(
            mix_peer - torch.minimum(global_per, local_per) + 1e-4
        ).mean()
    else:
        mix_regret = candidate_logits.new_zeros(())
    total = (
        weights.get("reason_final", 1.0) * final
        + weights.get("reason_global", 0.40) * global_view
        + weights.get("reason_local", 0.40) * local
        + weights.get("reason_rank", 0.05) * rank
        + weights.get("reason_soft_f1", 0.05) * soft_f1
        + weights.get("reason_annotation_delta", 0.02) * annotation_delta
        + weights.get("reason_mix_regret", 0.10) * mix_regret
    )
    return {
        "final": final,
        "global": global_view,
        "local": local,
        "rank": rank,
        "soft_f1": soft_f1,
        "annotation_delta": annotation_delta,
        "mix_regret": mix_regret,
        "total": total,
    }


def meter_reason_share_loss(
    output: dict[str, Tensor],
    target: Tensor,
    confidence: Tensor,
    factor_id: int,
    weights: dict[str, float] | None = None,
    *,
    observability: Tensor | None = None,
) -> dict[str, Tensor]:
    """Factor-local reason objective used only by the virtual meta update."""
    if factor_id < 0 or factor_id >= target.shape[1]:
        raise IndexError(f"factor id out of range: {factor_id}")
    weights = weights or {}
    column = slice(factor_id, factor_id + 1)
    target_r = target[:, column]
    confidence_r = confidence[:, column]
    observability_r = None if observability is None else observability[:, column]
    candidate = output.get("reason_logits_candidate", output["reason_logits_final"])[:, column]
    final = weighted_reason_asl(candidate, target_r, confidence_r, observability_r)
    global_view = weighted_reason_asl(
        output["reason_logits_global"][:, column],
        target_r,
        confidence_r,
        observability_r,
    )
    local = weighted_reason_asl(
        output["reason_logits_local"][:, column],
        target_r,
        confidence_r,
        observability_r,
    )
    soft_f1 = reason_soft_f1(candidate, target_r, confidence_r, observability_r)
    annotation_delta = output["reason_annotation_delta"][:, column].square().mean()
    total = (
        weights.get("reason_final", 1.0) * final
        + weights.get("reason_global", 0.40) * global_view
        + weights.get("reason_local", 0.40) * local
        + weights.get("reason_soft_f1", 0.05) * soft_f1
        + weights.get("reason_annotation_delta", 0.02) * annotation_delta
    )
    return {
        "final": final,
        "global": global_view,
        "local": local,
        "soft_f1": soft_f1,
        "annotation_delta": annotation_delta,
        "total": total,
    }
