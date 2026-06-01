from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class GradNormStats:
    action_weight: float
    reason_weight: float
    relation_weight: float
    min_weight: float
    max_weight: float


class GradNormBalancer(nn.Module):
    """Lightweight trainable task weighting with bounded weights."""

    def __init__(self, min_weight: float = 0.7, max_weight: float = 1.5) -> None:
        super().__init__()
        self.log_w_action = nn.Parameter(torch.zeros(()))
        self.log_w_reason = nn.Parameter(torch.zeros(()))
        self.log_w_relation = nn.Parameter(torch.zeros(()))
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)

    def _weight(self, param: torch.Tensor) -> torch.Tensor:
        return torch.exp(param).clamp(self.min_weight, self.max_weight)

    def forward(self, losses: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        wa = self._weight(self.log_w_action)
        wr = self._weight(self.log_w_reason)
        wt = self._weight(self.log_w_relation)
        total = wa * losses.get("action", torch.tensor(0.0, device=wa.device))
        total = total + wr * losses.get("reason", torch.tensor(0.0, device=wa.device))
        total = total + wt * losses.get("relation_teacher", torch.tensor(0.0, device=wa.device))
        stats = GradNormStats(float(wa.detach().cpu()), float(wr.detach().cpu()), float(wt.detach().cpu()), self.min_weight, self.max_weight)
        return total, stats.__dict__
