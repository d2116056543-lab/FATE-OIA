"""P12 task losses for multi-label RAEL action/reason supervision.

All public losses use logits and independent sigmoid probabilities.  None of
them introduce a softmax competition across labels.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from torch.nn import functional as F


Reduction = Literal["mean", "sum", "none"]
_SAFE_LOGIT_ABS = 1.0e6


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError("reduction must be one of: mean, sum, none")


def _validate_pair(
    logits: Tensor,
    target: Tensor,
    *,
    target_name: str = "target",
    validate_values: bool = False,
    require_binary_target: bool = False,
) -> None:
    if logits.ndim != 2 or target.ndim != 2:
        raise ValueError("logits and target must have shape [B,K]")
    if logits.shape != target.shape:
        raise ValueError("logits and target shape mismatch")
    if logits.shape[0] < 1 or logits.shape[1] < 1:
        raise ValueError("logits and target require nonempty [B,K]")
    if not torch.is_floating_point(logits) or not torch.is_floating_point(target):
        raise TypeError("logits and target must be floating point")
    if logits.device != target.device:
        raise ValueError("logits and target must share a device")
    if validate_values:
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(target).all()):
            raise ValueError("logits and target must be finite")
        if not bool((logits.float().abs() <= _SAFE_LOGIT_ABS).all()):
            raise ValueError(f"logits must satisfy abs(logits) <= {_SAFE_LOGIT_ABS:g} in debug validation")
        if not bool(((target >= 0.0) & (target <= 1.0)).all()):
            raise ValueError(f"{target_name} must be in [0,1]")
        if require_binary_target and not bool(((target == 0.0) | (target == 1.0)).all()):
            raise ValueError(f"{target_name} must be exactly binary in debug validation")


def _validate_logits_pair(first_logits: Tensor, second_logits: Tensor, *, validate_values: bool = False) -> None:
    """Validate two logit tensors without imposing target probability bounds."""
    if first_logits.ndim != 2 or second_logits.ndim != 2:
        raise ValueError("first_logits and second_logits must have shape [B,K]")
    if first_logits.shape != second_logits.shape:
        raise ValueError("first_logits and second_logits shape mismatch")
    if first_logits.shape[0] < 1 or first_logits.shape[1] < 1:
        raise ValueError("first_logits and second_logits require nonempty [B,K]")
    if not torch.is_floating_point(first_logits) or not torch.is_floating_point(second_logits):
        raise TypeError("first_logits and second_logits must be floating point")
    if first_logits.device != second_logits.device:
        raise ValueError("first_logits and second_logits must share a device")
    if validate_values:
        if not bool(torch.isfinite(first_logits).all()) or not bool(torch.isfinite(second_logits).all()):
            raise ValueError("first_logits and second_logits must be finite")
        if not bool(((first_logits.float().abs() <= _SAFE_LOGIT_ABS) & (second_logits.float().abs() <= _SAFE_LOGIT_ABS)).all()):
            raise ValueError(f"logits must satisfy abs(logits) <= {_SAFE_LOGIT_ABS:g} in debug validation")


def _reduce(loss: Tensor, reduction: Reduction) -> Tensor:
    _validate_reduction(reduction)
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    return loss.mean()


def multilabel_asymmetric_loss(
    logits: Tensor,
    target: Tensor,
    *,
    gamma_negative: float = 4.0,
    gamma_positive: float = 0.0,
    clip: float = 0.05,
    reduction: Reduction = "mean",
    validate_values: bool = False,
) -> Tensor:
    """Numerically stable asymmetric sigmoid loss for independent labels."""
    _validate_pair(logits, target, validate_values=validate_values, require_binary_target=True)
    if gamma_negative < 0.0 or gamma_positive < 0.0 or clip < 0.0 or clip >= 1.0:
        raise ValueError("ASL gamma values must be nonnegative and clip in [0,1)")
    logit_fp32 = logits.float()
    target_fp32 = target.float()
    positive_probability = torch.sigmoid(logit_fp32)
    # This exactly follows asymmetric_loss.py.  For a negative target the
    # focusing base is 1 - clipped_negative = (p_positive - clip)_+.
    negative_probability = 1.0 - positive_probability
    if clip > 0.0:
        negative_probability = (negative_probability + clip).clamp(max=1.0)
    epsilon = 1e-8
    positive_term = target_fp32 * torch.log(positive_probability.clamp_min(epsilon))
    negative_term = (1.0 - target_fp32) * torch.log(negative_probability.clamp_min(epsilon))
    if gamma_positive > 0.0 or gamma_negative > 0.0:
        point_probability = positive_probability * target_fp32 + negative_probability * (1.0 - target_fp32)
        gamma = gamma_positive * target_fp32 + gamma_negative * (1.0 - target_fp32)
        focusing = (1.0 - point_probability).clamp_min(epsilon).pow(gamma)
        positive_term = positive_term * focusing
        negative_term = negative_term * focusing
    return _reduce(-(positive_term + negative_term), reduction)


def soft_f1_loss(
    logits: Tensor,
    target: Tensor,
    *,
    epsilon: float = 1e-6,
    reduction: Reduction = "mean",
    validate_values: bool = False,
) -> Tensor:
    """Independent per-label soft-F1 loss, reduced only after label scoring."""
    _validate_pair(logits, target, validate_values=validate_values, require_binary_target=True)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    probabilities = torch.sigmoid(logits.float())
    target_fp32 = target.float()
    true_positive = (probabilities * target_fp32).sum(dim=0)
    false_positive = (probabilities * (1.0 - target_fp32)).sum(dim=0)
    false_negative = ((1.0 - probabilities) * target_fp32).sum(dim=0)
    loss_by_label = 1.0 - (2.0 * true_positive + epsilon) / (2.0 * true_positive + false_positive + false_negative + epsilon)
    return _reduce(loss_by_label, reduction)


def two_way_consistency_loss(
    first_logits: Tensor,
    second_logits: Tensor,
    *,
    reduction: Reduction = "mean",
    validate_values: bool = False,
) -> Tensor:
    """Symmetric stop-gradient Bernoulli agreement for two independent heads."""
    _validate_logits_pair(first_logits, second_logits, validate_values=validate_values)
    first = first_logits.float()
    second = second_logits.float()
    first_target = torch.sigmoid(first).detach()
    second_target = torch.sigmoid(second).detach()
    forward = F.binary_cross_entropy_with_logits(first, second_target, reduction="none")
    backward = F.binary_cross_entropy_with_logits(second, first_target, reduction="none")
    return _reduce(0.5 * (forward + backward), reduction)


def multilabel_ranking_loss(
    logits: Tensor,
    target: Tensor,
    *,
    margin: float = 0.25,
    reduction: Reduction = "mean",
    validate_values: bool = False,
) -> Tensor:
    """Hardest-positive versus hardest-negative margin loss for multi-label rows."""
    _validate_pair(logits, target, validate_values=validate_values, require_binary_target=True)
    if margin < 0.0:
        raise ValueError("margin must be nonnegative")
    scores = logits.float()
    positive = target > 0.5
    negative = ~positive
    valid = positive.any(dim=1) & negative.any(dim=1)
    per_sample = scores.new_zeros(scores.shape[0])
    weakest_positive = scores.masked_fill(~positive, float("inf")).min(dim=1).values
    strongest_negative = scores.masked_fill(~negative, float("-inf")).max(dim=1).values
    per_sample = torch.relu(margin - weakest_positive + strongest_negative)
    per_sample = torch.where(valid, per_sample, torch.zeros_like(per_sample))
    if reduction == "none":
        return per_sample
    denominator = valid.float().sum().clamp_min(1.0)
    if reduction == "sum":
        return per_sample.sum()
    _validate_reduction(reduction)
    return per_sample.sum() / denominator


def evidence_conditional_loss(
    logits: Tensor,
    soft_target: Tensor,
    positive_weight: Tensor,
    negative_weight: Tensor,
    *,
    reduction: Reduction = "mean",
    validate_values: bool = False,
) -> Tensor:
    """Weighted Bernoulli loss for P12 soft labels without hard-zero negatives."""
    _validate_pair(logits, soft_target, target_name="soft_target", validate_values=validate_values)
    for name, weight in (("positive_weight", positive_weight), ("negative_weight", negative_weight)):
        if weight.shape != logits.shape or weight.ndim != 2:
            raise ValueError(f"{name} shape mismatch")
        if not torch.is_floating_point(weight) or weight.device != logits.device:
            raise ValueError(f"{name} must be floating point on the logits device")
        if validate_values:
            if not bool(torch.isfinite(weight).all()) or not bool((weight >= 0.0).all()):
                raise ValueError(f"{name} must be finite and nonnegative")
    score = logits.float()
    target = soft_target.float()
    pos_weight = positive_weight.float()
    neg_weight = negative_weight.float()
    per_label = pos_weight * target * F.softplus(-score) + neg_weight * (1.0 - target) * F.softplus(score)
    return _reduce(per_label, reduction)


__all__ = [
    "evidence_conditional_loss",
    "multilabel_asymmetric_loss",
    "multilabel_ranking_loss",
    "soft_f1_loss",
    "two_way_consistency_loss",
]
