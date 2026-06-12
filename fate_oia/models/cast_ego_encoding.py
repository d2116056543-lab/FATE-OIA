from __future__ import annotations

import torch
from torch import nn


class EgoPatchCoordinateEncoder(nn.Module):
    def __init__(self, dim: int, grid_hw: tuple[int, int]):
        super().__init__()
        self.dim = int(dim)
        self.grid_hw = tuple(grid_hw)
        self.proj = nn.Linear(8, dim)

    def build_ego_features(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h, w = self.grid_hw
        y = torch.linspace(0.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        x_norm = xx.reshape(-1)
        y_norm = yy.reshape(-1)
        front_center = torch.exp(-(((x_norm - 0.5) ** 2) / 0.08 + ((y_norm - 0.75) ** 2) / 0.20))
        left_corridor = torch.sigmoid((0.45 - x_norm) * 10) * torch.sigmoid((y_norm - 0.35) * 10)
        right_corridor = torch.sigmoid((x_norm - 0.55) * 10) * torch.sigmoid((y_norm - 0.35) * 10)
        upper_control_region = torch.sigmoid((0.45 - y_norm) * 10)
        return torch.stack(
            [
                x_norm,
                y_norm,
                torch.abs(x_norm - 0.5),
                y_norm,
                front_center,
                left_corridor,
                right_corridor,
                upper_control_region,
            ],
            dim=-1,
        )

    def forward(self, patch_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # patch_tokens: [B, S, N, D]
        ego = self.build_ego_features(patch_tokens.device, patch_tokens.dtype)
        if ego.shape[0] != patch_tokens.shape[2]:
            raise ValueError(f"ego grid has {ego.shape[0]} tokens, got patch token count {patch_tokens.shape[2]}")
        delta = self.proj(ego).view(1, 1, ego.shape[0], self.dim)
        return patch_tokens + delta, ego
