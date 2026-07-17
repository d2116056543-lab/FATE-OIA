"""Target rereader seeded by typed factor samples rather than coarse masks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MOSAICFactorSeededRereader(nn.Module):
    def __init__(self, *, dim: int, target_count: int = 4, grid_hw: tuple[int, int] = (45, 80)) -> None:
        super().__init__()
        self.dim = int(dim)
        self.target_count = int(target_count)
        self.grid_hw = tuple(grid_hw)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.sample_proj = nn.Linear(dim, dim, bias=False)
        self.offset_raw = nn.Parameter(torch.zeros(target_count, 2))
        self.norm = nn.LayerNorm(dim)
        self.support_head = nn.Linear(dim, 1, bias=False)
        self.veto_head = nn.Linear(dim, 1, bias=False)

    @property
    def max_local_offset(self) -> float:
        return 0.08

    def forward(
        self,
        feature_map: torch.Tensor,
        target_queries: torch.Tensor,
        sampling_coordinates: torch.Tensor,
        sampled_features: torch.Tensor,
        sample_attention: torch.Tensor,
        factor_to_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, d, height, width = feature_map.shape
        if d != self.dim or target_queries.shape != (b, self.target_count, d):
            raise ValueError("factor rereader target shape mismatch")
        if sampling_coordinates.ndim != 6 or sampling_coordinates.shape[-1] != 2:
            raise ValueError("factor rereader requires typed coordinates")
        if sampled_features.shape[:5] != sampling_coordinates.shape[:5] or sampled_features.shape[-1] != d:
            raise ValueError("factor rereader sampled feature shape mismatch")
        if sample_attention.shape != sampling_coordinates.shape[:-1]:
            raise ValueError("factor rereader attention shape mismatch")
        if factor_to_target.shape != (b, sampling_coordinates.shape[1], self.target_count):
            raise ValueError("factor rereader route shape mismatch")
        weights = sample_attention.clamp_min(0.0).reshape(b, sampling_coordinates.shape[1], -1)
        coords = sampling_coordinates.reshape(b, sampling_coordinates.shape[1], -1, 2)
        samples = sampled_features.reshape(b, sampling_coordinates.shape[1], -1, d)
        denom = weights.sum(-1, keepdim=True).clamp_min(1e-6)
        # Keep the denominator as [B,F,1]. Adding another singleton here
        # changes [B,F,2] into an unintended four-dimensional broadcast.
        factor_coords = (coords * weights[..., None]).sum(-2) / denom
        factor_features = (samples * weights[..., None]).sum(-2) / denom
        route = factor_to_target.clamp_min(0.0)
        route = route / route.sum(1, keepdim=True).clamp_min(1e-6)
        target_coords = torch.einsum("bft,bfc->btc", route, factor_coords).clamp(-1.0, 1.0)
        target_seed = torch.einsum("bft,bfd->btd", route, factor_features)
        local_offsets = self.max_local_offset * torch.tanh(self.offset_raw).view(1, self.target_count, 2)
        target_coords = (target_coords + local_offsets).clamp(-1.0, 1.0)
        grid = target_coords.view(b, self.target_count, 1, 2)
        sampled_map = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=False)
        sampled_map = sampled_map.squeeze(-1).transpose(1, 2)
        nodes = self.norm(sampled_map + self.sample_proj(target_seed) + self.query_proj(target_queries))
        support_logits = F.softplus(self.support_head(nodes).squeeze(-1))
        veto_logits = F.softplus(self.veto_head(nodes).squeeze(-1))
        return {
            "target_nodes": nodes,
            "target_coordinates": target_coords,
            "target_local_offsets": local_offsets.expand(b, -1, -1),
            "support_logits": support_logits,
            "veto_logits": veto_logits,
        }
