from __future__ import annotations

import torch
from torch import nn


class ReasonReliabilityGate(nn.Module):
    def __init__(self, reason_dim: int = 21) -> None:
        super().__init__()
        self.reason_dim = int(reason_dim)
        self.freq_bias = nn.Parameter(torch.zeros(reason_dim))

    def forward(self, reason_logits: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        uncertainty = 1.0 - (torch.sigmoid(reason_logits) - 0.5).abs() * 2.0
        return torch.sigmoid(support + uncertainty + self.freq_bias)
