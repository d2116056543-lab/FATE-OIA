from __future__ import annotations

import torch
from torch import nn


class _LaneAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int, scale: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.up(self.act(self.down(self.norm(x))))


class TFCDualLaneAdapter(nn.Module):
    def __init__(self, dim: int = 384, bottleneck: int = 64, scale: float = 0.1) -> None:
        super().__init__()
        self.action_adapter = _LaneAdapter(dim, bottleneck, scale)
        self.reason_adapter = _LaneAdapter(dim, bottleneck, scale)

    def forward(self, patch_tokens_by_layer: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "patch_action": self.action_adapter(patch_tokens_by_layer),
            "patch_reason": self.reason_adapter(patch_tokens_by_layer),
        }
