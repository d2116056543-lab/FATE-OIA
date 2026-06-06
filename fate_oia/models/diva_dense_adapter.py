from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ActionSpecificLayerMixer(nn.Module):
    """Mix DINO layers separately for each action; never averages the action dimension."""

    def __init__(self, dim: int = 384, action_dim: int = 4, layer_indices=(3, 6, 9, 12)) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.layer_indices = tuple(int(x) for x in layer_indices)
        self.action_queries = nn.Parameter(torch.randn(action_dim, dim) * 0.02)
        self.layer_gate = nn.Linear(dim, len(self.layer_indices))

    def forward(self, maps_by_layer: dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        maps = [maps_by_layer[i] for i in self.layer_indices]
        stacked = torch.stack(maps, dim=1)  # [B,L,D,H,W]
        gates = torch.softmax(self.layer_gate(self.action_queries), dim=-1)  # [A,L]
        mixed = torch.einsum("al,bldhw->badhw", gates, stacked)
        return mixed.contiguous(), gates


class _LocalBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        groups = 8 if dim % 8 == 0 else 1
        self.depthwise = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.pointwise = nn.Conv2d(dim, dim, 1)
        self.norm = nn.GroupNorm(groups, dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.norm(self.pointwise(self.depthwise(x))))


class DrivingDenseAdapter(nn.Module):
    """Small driving-aware dense adapter producing P1/P2/P3 while preserving action axis."""

    def __init__(self, dim: int = 384, action_dim: int = 4, depth: int = 2) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.blocks = nn.Sequential(*[_LocalBlock(dim) for _ in range(depth)])
        self.p2 = nn.Conv2d(dim, dim, 1)
        self.p3 = nn.Conv2d(dim, dim, 1)

    def forward(self, action_maps: torch.Tensor) -> dict[str, torch.Tensor]:
        b, a, d, h, w = action_maps.shape
        x = action_maps.reshape(b * a, d, h, w)
        p1 = self.blocks(x).reshape(b, a, d, h, w)
        p2 = self.p2(F.adaptive_avg_pool2d(p1.reshape(b * a, d, h, w), (23, 40))).reshape(b, a, d, 23, 40)
        p3 = self.p3(F.adaptive_avg_pool2d(p1.reshape(b * a, d, h, w), (12, 20))).reshape(b, a, d, 12, 20)
        return {"P1": p1, "P2": p2, "P3": p3}
