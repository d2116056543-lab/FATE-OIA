from __future__ import annotations

import torch
from torch import nn


class ACPREgoRegionEncoder(nn.Module):
    def __init__(self, grid_hw: tuple[int, int] = (45, 80), dim: int = 384) -> None:
        super().__init__()
        self.grid_hw = grid_hw
        self.proj = nn.Linear(8, dim)

    def features(
        self, grid_hw: tuple[int, int] | None = None, device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        # Backward compatibility: old callers passed (device, dtype) positionally.
        if isinstance(grid_hw, torch.device):
            legacy_device = grid_hw
            legacy_dtype = device if isinstance(device, torch.dtype) else dtype
            device = legacy_device
            dtype = legacy_dtype
            grid_hw = None
        grid_hw = grid_hw or self.grid_hw
        device = device or torch.device("cpu")
        dtype = dtype or torch.float32
        h, w = grid_hw
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, h, device=device, dtype=dtype),
            torch.linspace(0, 1, w, device=device, dtype=dtype),
            indexing="ij",
        )
        front_center = torch.exp(-(((xx - 0.5) ** 2) / 0.08 + ((yy - 0.75) ** 2) / 0.20))
        left_corridor = torch.sigmoid((0.45 - xx) * 10.0) * torch.sigmoid((yy - 0.35) * 10.0)
        right_corridor = torch.sigmoid((xx - 0.55) * 10.0) * torch.sigmoid((yy - 0.35) * 10.0)
        upper_traffic_region = torch.sigmoid((0.45 - yy) * 10.0)
        bottom_drivable_region = yy
        ego_distance = torch.sqrt((xx - 0.5) ** 2 + (yy - 1.0) ** 2)
        feats = torch.stack(
            [xx, yy, front_center, left_corridor, right_corridor, upper_traffic_region, bottom_drivable_region, ego_distance],
            dim=-1,
        )
        return feats.view(h * w, 8)

    def forward(
        self, patch_tokens: torch.Tensor, grid_hw: tuple[int, int] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        b, n, d = patch_tokens.shape
        feats = self.features(grid_hw, patch_tokens.device, patch_tokens.dtype)
        if feats.shape[0] != n:
            raise ValueError(f"Expected {feats.shape[0]} patches, got {n}")
        ego_embed = self.proj(feats).unsqueeze(0).expand(b, -1, -1)
        region_masks = {
            "front_center": feats[:, 2],
            "left_corridor": feats[:, 3],
            "right_corridor": feats[:, 4],
            "upper_traffic_region": feats[:, 5],
            "bottom_drivable_region": feats[:, 6],
        }
        stats = {
            "front_center_mass": float(region_masks["front_center"].mean().detach().cpu()),
            "left_corridor_mass": float(region_masks["left_corridor"].mean().detach().cpu()),
            "right_corridor_mass": float(region_masks["right_corridor"].mean().detach().cpu()),
            "upper_traffic_region_mass": float(region_masks["upper_traffic_region"].mean().detach().cpu()),
            "bottom_drivable_region_mass": float(region_masks["bottom_drivable_region"].mean().detach().cpu()),
        }
        return patch_tokens + ego_embed, feats, region_masks, stats
