from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class EgoStats:
    left_corridor_mass: float
    right_corridor_mass: float
    front_center_mass: float
    upper_region_mass: float


class EaglePUEgoEncoding(nn.Module):
    feature_names = [
        "x_norm",
        "y_norm",
        "center_abs_x",
        "bottomness",
        "front_center",
        "left_corridor",
        "right_corridor",
        "upper_control_region",
    ]

    def __init__(self, grid_hw: tuple[int, int] = (45, 80), dim: int = 384) -> None:
        super().__init__()
        self.grid_hw = tuple(grid_hw)
        self.proj = nn.Linear(8, dim)

    def features(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, dict[str, float]]:
        h, w = self.grid_hw
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=dtype),
            torch.linspace(0, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        x = xx.reshape(-1)
        y = yy.reshape(-1)
        front_center = torch.exp(-(((x - 0.5) ** 2) / 0.08 + ((y - 0.75) ** 2) / 0.20))
        left_corridor = torch.sigmoid((0.45 - x) * 10) * torch.sigmoid((y - 0.35) * 10)
        right_corridor = torch.sigmoid((x - 0.55) * 10) * torch.sigmoid((y - 0.35) * 10)
        upper = torch.sigmoid((0.45 - y) * 10)
        feats = torch.stack([x, y, (x - 0.5).abs(), y, front_center, left_corridor, right_corridor, upper], dim=-1)
        stats = {
            "left_corridor_mass": float(left_corridor.mean().detach().cpu()),
            "right_corridor_mass": float(right_corridor.mean().detach().cpu()),
            "front_center_mass": float(front_center.mean().detach().cpu()),
            "upper_region_mass": float(upper.mean().detach().cpu()),
        }
        return feats, stats

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        feats, stats = self.features(patch_tokens.device, patch_tokens.dtype)
        return self.proj(feats).unsqueeze(0).expand(patch_tokens.shape[0], -1, -1), stats
