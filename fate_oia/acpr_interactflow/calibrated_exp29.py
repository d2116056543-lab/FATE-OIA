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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit label thresholds from train-only logits, never from test metrics."""
    logits = logits.float().detach().cpu()
    targets = targets.float().detach().cpu()
    mask = mask.float().detach().cpu()
    rates = positive_rate_targets(targets, mask, pi_min=pi_min, pi_max=pi_max)
    theta = torch.zeros(logits.shape[1], dtype=torch.float32)
    valid = mask > 0.5
    for label_idx in range(logits.shape[1]):
        label_logits = logits[valid[:, label_idx], label_idx]
        if label_logits.numel() == 0:
            continue
        theta[label_idx] = torch.quantile(label_logits, 1.0 - float(rates[label_idx].clamp(0.0, 1.0)))
    if deploy_logit_margin:
        theta = theta - float(deploy_logit_margin)
    return theta, rates
