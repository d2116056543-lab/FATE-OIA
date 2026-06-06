from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MultiScaleDeformableSampler(nn.Module):
    """Pure PyTorch grid_sample based multi-scale sampler."""

    def __init__(self, dim: int, num_scales: int = 3, num_points: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_scales = int(num_scales)
        self.num_points = int(num_points)
        self.offset = nn.Linear(dim, num_scales * num_points * 2)
        self.weight = nn.Linear(dim, num_scales * num_points)
        self.out = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, features: list[torch.Tensor], reference_points: torch.Tensor) -> dict[str, torch.Tensor]:
        b, q, d = query.shape
        if len(features) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} feature scales, got {len(features)}")
        offsets = torch.tanh(self.offset(query)).view(b, q, self.num_scales, self.num_points, 2) * 0.12
        weights = torch.softmax(self.weight(query).view(b, q, self.num_scales, self.num_points), dim=(-1))
        base = reference_points.view(b, q, 1, 1, 2)
        points = (base + offsets).clamp(0.0, 1.0)
        contexts = []
        for s, feat in enumerate(features):
            grid = points[:, :, s].reshape(b, q * self.num_points, 1, 2) * 2.0 - 1.0
            sampled = F.grid_sample(feat, grid, align_corners=False, mode="bilinear", padding_mode="border")
            sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, q, self.num_points, d)
            contexts.append(sampled)
        stacked = torch.stack(contexts, dim=2)  # [B,Q,S,K,D]
        context = (stacked * weights.unsqueeze(-1)).sum(dim=(2, 3))
        return {"context": self.out(context), "sample_points": points, "sample_weights": weights}
