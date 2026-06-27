from __future__ import annotations

import torch
from torch import nn


class ResponseLagEstimator(nn.Module):
    def __init__(self, dim: int = 384, max_lag: int = 4) -> None:
        super().__init__()
        self.max_lag = max_lag
        self.head = nn.Linear(dim, max_lag + 1)

    def forward(self, factor_tokens_trajectory: torch.Tensor, disabled: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if factor_tokens_trajectory.ndim != 4:
            raise ValueError(f"Expected [B,T,F,D], got {tuple(factor_tokens_trajectory.shape)}")
        b, t, f, d = factor_tokens_trajectory.shape
        last = factor_tokens_trajectory[:, -1]
        if disabled:
            weights = factor_tokens_trajectory.new_zeros(b, f, self.max_lag + 1)
            weights[..., 0] = 1.0
        else:
            weights = torch.softmax(self.head(last), dim=-1)
        lagged = []
        for lag in range(self.max_lag + 1):
            idx = max(0, t - 1 - lag)
            lagged.append(factor_tokens_trajectory[:, idx])
        lagged_tokens = torch.stack(lagged, dim=2)  # [B,F,L,D]
        context = (weights.unsqueeze(-1) * lagged_tokens).sum(2)
        return weights, context
