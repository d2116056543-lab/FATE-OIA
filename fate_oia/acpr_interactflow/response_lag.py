from __future__ import annotations

import torch
from torch import nn


class ResponseLagEstimator(nn.Module):
    def __init__(self, dim: int = 384, max_lag: int = 4) -> None:
        super().__init__()
        self.max_lag = max_lag
        self.head = nn.Linear(dim, max_lag + 1)
        self.lag_embedding = nn.Parameter(torch.randn(max_lag + 1, dim) * 0.02)

    def forward(self, motion_token: torch.Tensor, disabled: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if disabled:
            weights = motion_token.new_zeros(motion_token.shape[0], self.max_lag + 1)
            weights[:, 0] = 1.0
        else:
            weights = torch.softmax(self.head(motion_token), dim=-1)
        context = weights @ self.lag_embedding
        return weights, context
