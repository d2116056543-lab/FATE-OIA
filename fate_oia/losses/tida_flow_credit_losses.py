from __future__ import annotations

import torch
import torch.nn.functional as F


def signed_gt_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError("logits and target must have identical shape")
    return (2.0 * target.to(logits.dtype) - 1.0) * logits


def conditional_credit_weight(need: torch.Tensor) -> torch.Tensor:
    """Emphasize paired credit where the label-independent utility predicts need."""
    return (0.25 + 0.75 * need.detach().clamp(0.0, 1.0)).to(need.dtype)


def conditional_no_harm_weight(need: torch.Tensor) -> torch.Tensor:
    """Protect the image fallback most strongly when temporal utility is low."""
    return (1.0 - 0.75 * need.detach().clamp(0.0, 1.0)).to(need.dtype)


def temporal_utility_calibration_loss(
    budget: torch.Tensor,
    real_logits: torch.Tensor,
    counterfactual_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    max_budget: float = 1.0,
    temperature: float = 0.20,
) -> torch.Tensor:
    """Fit temporal budget to detached, target-signed paired benefit."""
    if budget.shape != real_logits.shape or real_logits.shape != counterfactual_logits.shape:
        raise ValueError("budget and paired logits must have identical shapes")
    benefit = (
        signed_gt_margin(real_logits, target) - signed_gt_margin(counterfactual_logits, target)
    ).detach()
    soft_target = float(max_budget) * torch.sigmoid(benefit / float(temperature))
    return F.smooth_l1_loss(budget, soft_target)


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    if weight is None:
        return value.mean()
    if weight.shape != value.shape:
        raise ValueError("sample_weight must match logits")
    weight = weight.detach().to(value.dtype)
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def counterfactual_margin_credit_loss(
    real_logits: torch.Tensor,
    counterfactual_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    margin: float = 0.02,
) -> torch.Tensor:
    real_margin = signed_gt_margin(real_logits, target)
    # Backpropagate through both members of the same-image pair. Shared static
    # terms then cancel, so the hinge assigns credit to the history-sensitive
    # path instead of merely raising the real branch's static logits.
    counterfactual_margin = signed_gt_margin(counterfactual_logits, target)
    return _weighted_mean(F.relu(counterfactual_margin - real_margin + float(margin)), sample_weight)


def image_fallback_no_harm_loss(
    image_logits: torch.Tensor,
    video_logits: torch.Tensor,
    target: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
    epsilon: float = 0.0,
) -> torch.Tensor:
    image_margin = signed_gt_margin(image_logits.detach(), target)
    video_margin = signed_gt_margin(video_logits, target)
    return _weighted_mean(F.relu(image_margin - video_margin + float(epsilon)), sample_weight)


def transition_alignment_loss(
    transition_tokens: torch.Tensor,
    predicate_innovation: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    if transition_tokens.shape != predicate_innovation.shape:
        raise ValueError("transition and predicate innovation shapes must match")
    if reliability.shape != transition_tokens.shape[:2]:
        raise ValueError("reliability must be [B,P]")
    prediction = F.layer_norm(transition_tokens, (transition_tokens.shape[-1],))
    target = F.layer_norm(predicate_innovation.detach(), (predicate_innovation.shape[-1],))
    error = F.smooth_l1_loss(prediction, target, reduction="none").mean(-1)
    weight = reliability.detach().clamp(0.0, 1.0)
    return (error * weight).sum() / weight.sum().clamp_min(1e-8)
