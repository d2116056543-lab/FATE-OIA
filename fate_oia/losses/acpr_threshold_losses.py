from __future__ import annotations

import torch
import torch.nn.functional as F


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weights = weights.to(values.device, values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def soft_f1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    tau: float = 0.25,
    label_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
    macro: bool = True,
) -> torch.Tensor:
    probs = torch.sigmoid(logits / float(tau))
    targets = targets.float()
    tp = (probs * targets).sum(0)
    denom = probs.sum(0) + targets.sum(0)
    f1 = (2.0 * tp + eps) / (denom + eps)
    if macro:
        return 1.0 - _weighted_mean(f1, label_weights)
    return 1.0 - (2.0 * tp.sum() + eps) / (denom.sum() + eps)


def predicted_positive_rate_loss(
    logits: torch.Tensor,
    target_rate: torch.Tensor,
    tau: float = 0.25,
    label_mask: torch.Tensor | None = None,
    kind: str = "smooth_l1",
) -> torch.Tensor:
    pred_rate = torch.sigmoid(logits / float(tau)).mean(0)
    target_rate = target_rate.to(pred_rate.device, pred_rate.dtype)
    if label_mask is not None:
        mask = label_mask.to(pred_rate.device).bool()
        pred_rate = pred_rate[mask]
        target_rate = target_rate[mask]
    if pred_rate.numel() == 0:
        return logits.sum() * 0.0
    if kind == "mse":
        return F.mse_loss(pred_rate, target_rate)
    return F.smooth_l1_loss(pred_rate, target_rate)


def action_cardinality_loss(action_logits_deploy: torch.Tensor, action_targets: torch.Tensor, tau: float = 0.25) -> torch.Tensor:
    soft_count = torch.sigmoid(action_logits_deploy / float(tau)).sum(-1)
    target_count = action_targets.float().sum(-1)
    return F.smooth_l1_loss(soft_count, target_count)


def threshold_teacher_loss(
    threshold_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    label_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(threshold_logit, teacher_logit.to(threshold_logit.device, threshold_logit.dtype), reduction="none")
    return _weighted_mean(loss, label_weights)


def threshold_prior_loss(
    threshold_logit: torch.Tensor,
    prior_logit: torch.Tensor,
    label_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(threshold_logit, prior_logit.to(threshold_logit.device, threshold_logit.dtype), reduction="none")
    return _weighted_mean(loss, label_weights)


def threshold_range_penalty(threshold_prob: torch.Tensor, min_prob: torch.Tensor, max_prob: torch.Tensor) -> torch.Tensor:
    min_prob = min_prob.to(threshold_prob.device, threshold_prob.dtype)
    max_prob = max_prob.to(threshold_prob.device, threshold_prob.dtype)
    return (F.relu(min_prob - threshold_prob) + F.relu(threshold_prob - max_prob)).mean()


def deploy_fp_guard_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    tau: float = 0.25,
    max_fp_rate: torch.Tensor | None = None,
) -> torch.Tensor:
    if max_fp_rate is None:
        return logits.sum() * 0.0
    probs = torch.sigmoid(logits / float(tau))
    fp_rate = (probs * (1.0 - targets.float())).mean(0)
    max_fp_rate = max_fp_rate.to(fp_rate.device, fp_rate.dtype)
    return F.relu(fp_rate - max_fp_rate).mean()


def calalign_loss_bundle(
    action_logits_deploy: torch.Tensor,
    reason_logits_deploy: torch.Tensor,
    action_targets: torch.Tensor,
    reason_targets: torch.Tensor,
    threshold_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    prior_logit: torch.Tensor,
    target_rate: torch.Tensor,
    tau: float = 0.25,
    threshold_prob: torch.Tensor | None = None,
    min_prob: torch.Tensor | None = None,
    max_prob: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or {}
    deploy = torch.cat([action_logits_deploy, reason_logits_deploy], dim=-1)
    targets = torch.cat([action_targets, reason_targets], dim=-1)
    losses = {
        "loss_threshold_soft_f1_action": soft_f1_loss(action_logits_deploy, action_targets, tau=tau),
        "loss_threshold_soft_f1_reason": soft_f1_loss(reason_logits_deploy, reason_targets, tau=tau),
        "loss_threshold_rate": predicted_positive_rate_loss(deploy, target_rate, tau=tau),
        "loss_action_cardinality": action_cardinality_loss(action_logits_deploy, action_targets, tau=tau),
        "loss_threshold_teacher": threshold_teacher_loss(threshold_logit, teacher_logit),
        "loss_threshold_prior": threshold_prior_loss(threshold_logit, prior_logit),
        "loss_threshold_range": threshold_range_penalty(threshold_prob, min_prob, max_prob)
        if threshold_prob is not None and min_prob is not None and max_prob is not None
        else threshold_logit.sum() * 0.0,
    }
    losses["total"] = (
        float(weights.get("soft_f1_action", 1.0)) * losses["loss_threshold_soft_f1_action"]
        + float(weights.get("soft_f1_reason", 1.0)) * losses["loss_threshold_soft_f1_reason"]
        + float(weights.get("rate", 1.0)) * losses["loss_threshold_rate"]
        + float(weights.get("action_cardinality", 1.0)) * losses["loss_action_cardinality"]
        + float(weights.get("teacher", 1.0)) * losses["loss_threshold_teacher"]
        + float(weights.get("prior", 1.0)) * losses["loss_threshold_prior"]
        + float(weights.get("range", 1.0)) * losses["loss_threshold_range"]
    )
    return losses
