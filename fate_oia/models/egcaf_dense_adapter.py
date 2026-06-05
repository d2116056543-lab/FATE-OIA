from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class EGCafLayerProjector(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class EGCafFeaturePyramid(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

    def forward(self, p1: torch.Tensor) -> dict[str, torch.Tensor]:
        p1 = p1 + self.local(p1)
        return {
            "P1": p1,
            "P2": F.avg_pool2d(p1, 2, 2, ceil_mode=True),
            "P3": F.avg_pool2d(p1, 4, 4, ceil_mode=True),
        }


class EGCafDrivingDenseAdapter(nn.Module):
    def __init__(self, input_dim: int = 384, hidden_dim: int = 256, num_actions: int = 4, layer_names: list[str] | None = None) -> None:
        super().__init__()
        self.layer_names = layer_names or ["layer_3", "layer_6", "layer_9", "layer_12"]
        self.projectors = nn.ModuleDict({name: EGCafLayerProjector(input_dim, hidden_dim) for name in self.layer_names})
        self.action_embed = nn.Parameter(torch.randn(num_actions, hidden_dim) * 0.02)
        self.layer_gate = nn.Linear(hidden_dim, len(self.layer_names))
        self.fpn = EGCafFeaturePyramid(hidden_dim)

    def forward(self, layer_tokens: dict[str, torch.Tensor], grid_hw: tuple[int, int]) -> dict[str, Any]:
        h, w = grid_hw
        maps = []
        stats = {}
        for name in self.layer_names:
            tok = self.projectors[name](layer_tokens[name])
            maps.append(tok.transpose(1, 2).reshape(tok.shape[0], tok.shape[2], h, w))
            stats[f"{name}_mean"] = float(tok.detach().mean().cpu())
            stats[f"{name}_std"] = float(tok.detach().std().cpu())
        stack = torch.stack(maps, dim=1)
        gates = torch.softmax(self.layer_gate(self.action_embed), dim=-1)
        action_maps = []
        for a in range(gates.shape[0]):
            action_maps.append(self.fpn((stack * gates[a].view(1, -1, 1, 1, 1)).sum(1)))
        return {"pyramid": action_maps, "layer_gates": gates, "dense_stats": stats, "base_p1": stack.mean(1)}
