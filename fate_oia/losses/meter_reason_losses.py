from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ReasonSupervision:
    positive_mask: Tensor
    unknown_mask: Tensor
    positive_weight: Tensor
    negative_weight: Tensor
    soft_positive_weight: Tensor


def noisy_zero_trust(
    ema_probability: Tensor,
    positive_state_probability: Tensor,
    reliability: Tensor,
    view_consistency: Tensor,
) -> tuple[Tensor, Tensor]:
    trust = (
        ema_probability.detach().clamp(0, 1)
        * positive_state_probability.detach().clamp(0, 1)
        * reliability.detach().clamp(0, 1)
        * view_consistency.detach().clamp(0, 1)
    )
    return trust, (1.0 - trust).clamp(0.10, 1.0)


def cross_view_consistency(
    logits: Tensor,
    view_logits: Tensor,
    measurement: Tensor,
    view_measurement: Tensor,
    *,
    alpha: float = 1.0,
    temperature: float = 1.0,
) -> Tensor:
    distance = (logits - view_logits).abs() + float(alpha) * (
        measurement - view_measurement
    ).abs()
    return torch.exp(-distance / max(float(temperature), 1e-6))


def build_reason_supervision(
    target: Tensor,
    missing_positive_trust: Tensor,
    *,
    soft_positive_weight: Tensor | None = None,
) -> ReasonSupervision:
    positive = target.gt(0.5)
    unknown = ~positive
    negative_weight = (1.0 - missing_positive_trust.detach()).clamp(0.10, 1.0)
    soft = (
        torch.zeros_like(target)
        if soft_positive_weight is None
        else soft_positive_weight.detach().clamp(0, 1) * unknown.to(target)
    )
    return ReasonSupervision(
        positive_mask=positive,
        unknown_mask=unknown,
        positive_weight=torch.ones_like(target),
        negative_weight=negative_weight,
        soft_positive_weight=soft,
    )


def robust_reason_asl(logits: Tensor, supervision: ReasonSupervision) -> Tensor:
    probability = torch.sigmoid(logits)
    positive_target = supervision.positive_mask.to(logits)
    soft = supervision.soft_positive_weight.to(logits)
    positive = -(positive_target + soft) * torch.log(probability.clamp_min(1e-6))
    negative = (
        -supervision.unknown_mask.to(logits)
        * (1.0 - soft)
        * supervision.negative_weight.to(logits)
        * probability.square()
        * torch.log((1.0 - probability).clamp_min(1e-6))
    )
    return (positive + negative).mean()


def robust_reason_soft_f1(logits: Tensor, supervision: ReasonSupervision) -> Tensor:
    probability = torch.sigmoid(logits)
    target = supervision.positive_mask.to(logits) + supervision.soft_positive_weight.to(logits)
    fp_weight = (
        supervision.unknown_mask.to(logits)
        * supervision.negative_weight.to(logits)
        * (1.0 - supervision.soft_positive_weight.to(logits))
    )
    tp = (probability * target).sum(0)
    fp = (probability * fp_weight).sum(0)
    fn = ((1.0 - probability) * target).sum(0)
    return 1.0 - ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean()


def robust_reason_rank_loss(
    logits: Tensor, supervision: ReasonSupervision, *, margin: float = 0.20
) -> Tensor:
    positive = supervision.positive_mask.to(logits) + supervision.soft_positive_weight.to(logits)
    negative = supervision.unknown_mask.to(logits) * supervision.negative_weight.to(logits)
    pair_weight = positive.unsqueeze(-1) * negative.unsqueeze(-2)
    if not bool(pair_weight.gt(0).any()):
        return logits.new_zeros(())
    loss = torch.relu(float(margin) - logits.unsqueeze(-1) + logits.unsqueeze(-2))
    return (loss * pair_weight).sum() / pair_weight.sum().clamp_min(1e-6)


def reason_correction_sign_loss(
    correction: Tensor,
    supervision: ReasonSupervision,
    *,
    margin: float = 0.05,
) -> Tensor:
    positive = supervision.positive_mask.to(correction)
    negative = supervision.unknown_mask.to(correction) * supervision.negative_weight.to(correction)
    terms: list[Tensor] = []
    if bool(positive.gt(0).any()):
        terms.append(
            (positive * torch.relu(float(margin) - correction)).sum()
            / positive.sum().clamp_min(1e-6)
        )
    if bool(negative.gt(0).any()):
        terms.append(
            (negative * torch.relu(float(margin) + correction)).sum()
            / negative.sum().clamp_min(1e-6)
        )
    return torch.stack(terms).mean() if terms else correction.new_zeros(())


def meter_reason_loss(
    output: dict[str, Tensor],
    target: Tensor,
    missing_positive_trust: Tensor,
    weights: dict[str, float] | None = None,
    *,
    soft_positive_weight: Tensor | None = None,
    view_output: dict[str, Tensor] | None = None,
) -> dict[str, Tensor | ReasonSupervision]:
    weights = weights or {}
    supervision = build_reason_supervision(
        target,
        missing_positive_trust,
        soft_positive_weight=soft_positive_weight,
    )
    final = robust_reason_asl(output["reason_logits_final"], supervision)
    global_loss = robust_reason_asl(output["reason_logits_global"], supervision)
    rank = robust_reason_rank_loss(output["reason_logits_final"], supervision)
    soft_f1 = robust_reason_soft_f1(output["reason_logits_final"], supervision)
    correction = reason_correction_sign_loss(
        output["reason_evidence_delta"], supervision
    )
    view = output["reason_logits_final"].new_zeros(())
    if view_output is not None:
        view = (
            1.0
            - cross_view_consistency(
                output["reason_logits_final"],
                view_output["reason_logits_final"],
                output["factor_reliability"],
                view_output["factor_reliability"],
            ).mean()
        )
    total = (
        weights.get("reason_final", 1.00) * final
        + weights.get("reason_global", 0.50) * global_loss
        + weights.get("reason_rank", 0.08) * rank
        + weights.get("reason_soft_f1", 0.05) * soft_f1
        + weights.get("reason_correction_sign", 0.08) * correction
        + weights.get("reason_view_consistency", 0.05) * view
    )
    return {
        "final": final,
        "global": global_loss,
        "rank": rank,
        "soft_f1": soft_f1,
        "correction_sign": correction,
        "view_consistency": view,
        "supervision": supervision,
        "total": total,
    }


def meter_reason_share_loss(*args: object, **kwargs: object) -> dict[str, Tensor]:
    raise RuntimeError("Formal meta-sharing is disabled in HECA")
