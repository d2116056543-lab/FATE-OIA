from __future__ import annotations

import torch
from torch import nn


class ResponseLagEstimator(nn.Module):
    def __init__(self, dim: int = 384, max_lag: int = 4) -> None:
        super().__init__()
        self.max_lag = max_lag
        self.head = nn.Linear(dim, max_lag + 1)

    def forward(self, motion_token: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.head(motion_token), dim=-1)

