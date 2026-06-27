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
