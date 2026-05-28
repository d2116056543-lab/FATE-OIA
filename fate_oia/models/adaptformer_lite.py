from __future__ import annotations

import torch
from torch import nn


class AdaptFormerLite(nn.Module):
    """Small bottleneck adapter kept disabled for the first ScoreV2 run."""

    def __init__(self, dim: int, bottleneck_dim: int = 64, dropout: float = 0.0, init_scale: float = 1e-3) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, dim),
        )
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.scale.to(dtype=tokens.dtype) * self.adapter(tokens)
