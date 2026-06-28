from __future__ import annotations

import torch
from torch import nn


class Exp29TrainOnlyCalibrator(nn.Module):
    """Train-only threshold/bias calibrator for fixed-threshold deployment."""

    def __init__(self, exp_dim: int = 29) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(exp_dim))
        self.delta = nn.Sequential(nn.Linear(exp_dim, exp_dim), nn.Tanh(), nn.Linear(exp_dim, exp_dim))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits - self.theta.view(1, -1) + 0.10 * self.delta(logits).clamp(-1.0, 1.0)


def positive_rate_targets(targets: torch.Tensor, mask: torch.Tensor, pi_min: float = 0.03, pi_max: float = 0.35) -> torch.Tensor:
    positives = ((targets > 0.5) & (mask > 0.5)).float()
    valid = (mask > 0.5).float()
    rate = positives.sum(0) / valid.sum(0).clamp_min(1.0)
    return rate.clamp(pi_min, pi_max)


def fit_exp29_theta_from_train_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pi_min: float = 0.03,
    pi_max: float = 0.35,
    deploy_logit_margin: float = 0.0,
    max_global_pred_rate: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit label thresholds from train-only logits, never from test metrics."""
    logits = logits.float().detach().cpu()
    targets = targets.float().detach().cpu()
    mask = mask.float().detach().cpu()
    rates = positive_rate_targets(targets, mask, pi_min=pi_min, pi_max=pi_max)
    theta = torch.zeros(logits.shape[1], dtype=torch.float32)
    # Use the full train-calib logit distribution to control deploy-time
    # positive rate. The mask still controls target priors and metric quality,
    # but unknown rows are not allowed to become unbounded positives simply
    # because they were excluded from threshold fitting.
    for label_idx in range(logits.shape[1]):
        label_logits = logits[:, label_idx]
        if label_logits.numel() == 0:
            continue
        theta[label_idx] = torch.quantile(label_logits, 1.0 - float(rates[label_idx].clamp(0.0, 1.0)))
    if deploy_logit_margin:
        theta = theta - float(deploy_logit_margin)
    if max_global_pred_rate is None:
        max_global_pred_rate = pi_max
    cap = float(max(0.0, min(1.0, max_global_pred_rate)))
    if cap > 0.0:
        for label_idx in range(logits.shape[1]):
            label_logits = logits[:, label_idx]
            if label_logits.numel() == 0:
                continue
            pred_rate = (label_logits >= theta[label_idx]).float().mean()
            if float(pred_rate) > cap:
                theta[label_idx] = torch.quantile(label_logits, 1.0 - cap) + 1e-4
    return theta, rates


def exp29_calibration_quality(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    theta: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Measure train-only deploy quality for a candidate theta.

    This intentionally uses only the supplied calibration tensors. It is used to
    prevent a later train-calib refit from replacing a usable theta with one that
    collapses fixed-threshold positive predictions.
    """
    logits = logits.float().detach().cpu()
    targets = targets.float().detach().cpu()
    mask = mask.float().detach().cpu()
    theta = theta.float().detach().cpu().view(1, -1)
    deploy = logits - theta
    probs = torch.sigmoid(deploy)
    pred_all = (probs >= threshold).float()
    valid = (mask > 0.5).float()
    target = (targets > 0.5).float() * valid
    pred = pred_all * valid
    tp = (pred * target).sum(0)
    fp = (pred * (1.0 - target) * valid).sum(0)
    fn = ((1.0 - pred) * target).sum(0)
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn).clamp_min(1e-8)
    valid_labels = (target.sum(0) > 0).float()
    mf1 = (f1 * valid_labels).sum() / valid_labels.sum().clamp_min(1.0)
    pred_positive_rate = pred_all.mean()
    valid_pred_positive_rate = pred.sum() / valid.sum().clamp_min(1.0)
    return {
        "mF1": float(mf1),
        "pred_positive_rate": float(pred_positive_rate),
        "valid_pred_positive_rate": float(valid_pred_positive_rate),
        "prob_mean": float(probs.mean()),
        "prob_max": float(probs.max()),
        "pred_cardinality_mean": float(pred.sum(-1).mean()),
    }


def should_accept_exp29_theta(
    candidate_quality: dict[str, float],
    current_quality: dict[str, float] | None,
    min_pred_positive_rate: float = 0.02,
    mf1_tolerance: float = 0.005,
) -> bool:
    """Accept a train-only theta refit only if it does not collapse deploy F1."""
    if candidate_quality.get("pred_positive_rate", 0.0) < min_pred_positive_rate:
        return False
    if current_quality is None:
        return True
    if current_quality.get("mF1", -1.0) < 0:
        return True
    return candidate_quality.get("mF1", -1.0) >= current_quality.get("mF1", -1.0) - mf1_tolerance
