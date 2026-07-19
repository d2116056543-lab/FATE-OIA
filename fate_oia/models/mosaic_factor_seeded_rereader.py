"""Target rereader seeded by typed factor samples rather than coarse masks."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MOSAICFactorSeededRereader(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        target_count: int = 4,
        grid_hw: tuple[int, int] = (45, 80),
        max_local_offset: float = 0.08,
        slot_count: int = 2,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.target_count = int(target_count)
        self.grid_hw = tuple(grid_hw)
        self.slot_count = int(slot_count)
        if not 0.0 < float(max_local_offset) <= 0.08:
            raise ValueError("fine rereader local offset must be in (0,0.08]")
        if self.slot_count < 1:
            raise ValueError("fine rereader slot_count must be positive")
        self._max_local_offset = float(max_local_offset)
        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.sample_proj = nn.Linear(dim, dim, bias=False)
        self.offset_proj = nn.Linear(2 * dim, 2, bias=True)
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        self.norm = nn.LayerNorm(dim)
        self.support_head = nn.Linear(dim, 1, bias=False)
        self.veto_head = nn.Linear(dim, 1, bias=False)

    @property
    def max_local_offset(self) -> float:
        return self._max_local_offset

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
        # ``factor_to_target`` is the mass-preserving weight w=m*pi, never a
        # pure attention distribution.  Keep m separate before selecting the
        # strongest independent factor slots; otherwise a weak route collapses
        # into a synthetic centroid and can produce a correction by itself.
        route = factor_to_target.clamp_min(0.0)
        route_mass = route.sum(1)
        zero_mass = route_mass <= 1e-8
        distribution = route / route_mass.unsqueeze(1).clamp_min(1e-8)
        distribution = distribution * (~zero_mass).unsqueeze(1).to(distribution.dtype)
        slot_count = min(self.slot_count, route.shape[1])
        slot_distribution, topk_factor_ids = distribution.transpose(1, 2).topk(slot_count, dim=-1)
        slot_distribution = slot_distribution / slot_distribution.sum(-1, keepdim=True).clamp_min(1e-8)
        factor_coords_by_target = factor_coords.unsqueeze(1).expand(-1, self.target_count, -1, -1)
        factor_features_by_target = factor_features.unsqueeze(1).expand(-1, self.target_count, -1, -1)
        coordinate_index = topk_factor_ids.unsqueeze(-1).expand(-1, -1, -1, 2)
        feature_index = topk_factor_ids.unsqueeze(-1).expand(-1, -1, -1, d)
        slot_coordinates = factor_coords_by_target.gather(2, coordinate_index)
        slot_seed = factor_features_by_target.gather(2, feature_index)
        query_by_slot = target_queries.unsqueeze(2).expand(-1, -1, slot_count, -1)
        local_offsets = self.max_local_offset * torch.tanh(self.offset_proj(torch.cat((query_by_slot, slot_seed), dim=-1)))
        slot_coordinates = (slot_coordinates + local_offsets).clamp(-1.0, 1.0)
        grid = slot_coordinates.reshape(b, self.target_count * slot_count, 1, 2)
        sampled_map = F.grid_sample(feature_map, grid, mode="bilinear", align_corners=False)
        sampled_map = sampled_map.squeeze(-1).transpose(1, 2).reshape(b, self.target_count, slot_count, d)
        slot_nodes_raw = self.norm(
            sampled_map + self.sample_proj(slot_seed) + self.query_proj(query_by_slot)
        )
        # Gate every typed input, local sample, and resulting correction with
        # absolute mass.  This is the structural zero-evidence-zero-effect
        # guarantee required by CREDO-MAP, not a post-hoc diagnostic.
        slot_nodes = slot_nodes_raw * route_mass.unsqueeze(-1).unsqueeze(-1)
        nodes = torch.einsum("btk,btkd->btd", slot_distribution, slot_nodes)
        support_logits = F.softplus(self.support_head(nodes).squeeze(-1)) * route_mass
        veto_logits = F.softplus(self.veto_head(nodes).squeeze(-1)) * route_mass
        topk_factor_ids = torch.where(
            zero_mass.unsqueeze(-1), torch.full_like(topk_factor_ids, -1), topk_factor_ids
        )
        slot_coordinates = slot_coordinates * (~zero_mass).unsqueeze(-1).unsqueeze(-1).to(slot_coordinates.dtype)
        return {
            "target_nodes": nodes,
            "target_coordinates": torch.einsum("btk,btkc->btc", slot_distribution, slot_coordinates),
            "target_local_offsets": local_offsets,
            "support_logits": support_logits,
            "veto_logits": veto_logits,
            "route_distribution": distribution,
            "route_mass": route_mass,
            "zero_mass_mask": zero_mass,
            "topk_factor_ids": topk_factor_ids,
            "slot_coordinates": slot_coordinates,
            "slot_nodes": slot_nodes,
        }
