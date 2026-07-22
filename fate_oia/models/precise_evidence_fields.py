from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PRECISEEvidenceFields(nn.Module):
    """Multi-part explicit evidence and unnamed latent visual slots."""

    def __init__(self, fields: list[dict[str, Any]], dim: int = 384, latent_slots: int = 6, latent_parts: int = 4, grid_hw: tuple[int, int] = (45, 80), reliability_tau: float = 0.20) -> None:
        super().__init__()
        self.fields = fields
        self.grid_hw = grid_hw
        self.reliability_tau = reliability_tau
        self.num_explicit = len(fields)
        self.latent_slots = latent_slots
        self.max_parts = max(int(field["num_parts"]) for field in fields)
        self.explicit_queries = nn.Parameter(torch.randn(self.num_explicit, dim) * 0.02)
        self.latent_queries = nn.Parameter(torch.randn(latent_slots, dim) * 0.02)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.presence = nn.Linear(dim, 1)
        self.observability = nn.Linear(dim, 1)
        self.state_heads = nn.ModuleList([nn.Linear(dim, max(1, len(field.get("state_schema", field.get("type_schema", []))))) for field in fields])
        self.part_offsets = nn.Parameter(torch.zeros(self.num_explicit, self.max_parts, 2))
        self.part_scales_raw = nn.Parameter(torch.zeros(self.num_explicit, self.max_parts, 2))
        self.positive_prototypes = nn.Parameter(torch.randn(self.num_explicit, dim) * 0.02)
        self.negative_prototypes = nn.Parameter(torch.randn(self.num_explicit, dim) * 0.02)
        self.register_buffer("view_consistency_ema", torch.full((self.num_explicit,), 0.75))
        self.register_buffer("part_count", torch.tensor([int(field["num_parts"]) for field in fields], dtype=torch.long))

    def _coordinates(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        height, width = self.grid_hw
        yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, height, device=device, dtype=dtype), torch.linspace(0.0, 1.0, width, device=device, dtype=dtype), indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(height * width, 2)

    def _attention(self, queries: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("ed,bnd->ben", queries, self.key(tokens)) / (tokens.shape[-1] ** 0.5)
        return torch.softmax(logits, dim=-1)

    def _derived_atoms(self, presence: torch.Tensor, state_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        p = torch.sigmoid(presence)
        state = torch.sigmoid(state_logits)
        name_to_idx = {field["name"]: idx for idx, field in enumerate(self.fields)}
        atom = {
            "traffic_light_visible": p[:, name_to_idx["traffic_light"]],
            "traffic_light_red": p[:, name_to_idx["traffic_light"]] * state[:, name_to_idx["traffic_light"], 0],
            "traffic_light_green": p[:, name_to_idx["traffic_light"]] * state[:, name_to_idx["traffic_light"], 1],
            "traffic_sign_visible": p[:, name_to_idx["traffic_sign"]],
            "front_vehicle_visible": p[:, name_to_idx["actor_center"]],
            "front_pedestrian_visible": p[:, name_to_idx["actor_center"]] * state[:, name_to_idx["actor_center"], 1],
            "front_rider_visible": p[:, name_to_idx["actor_center"]] * state[:, name_to_idx["actor_center"], 2],
            "front_other_obstacle": p[:, name_to_idx["actor_center"]] * state[:, name_to_idx["actor_center"], 3],
            "left_occupied": p[:, name_to_idx["actor_left"]],
            "center_occupied": p[:, name_to_idx["actor_center"]],
            "right_occupied": p[:, name_to_idx["actor_right"]],
            "left_drivable": p[:, name_to_idx["drivable_left"]],
            "center_drivable": p[:, name_to_idx["drivable_center"]],
            "right_drivable": p[:, name_to_idx["drivable_right"]],
            "left_boundary_visible": p[:, name_to_idx["boundary_left"]],
            "right_boundary_visible": p[:, name_to_idx["boundary_right"]],
            "left_solid_boundary": p[:, name_to_idx["boundary_left"]] * state[:, name_to_idx["boundary_left"], 0],
            "right_solid_boundary": p[:, name_to_idx["boundary_right"]] * state[:, name_to_idx["boundary_right"], 0],
        }
        return atom

    def forward(self, evidence_layers: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        batch, layer_count, token_count, dim = evidence_layers.shape
        if (layer_count, token_count, dim) != (3, 3600, 384):
            raise ValueError("PRECISE evidence receives the uncompressed three-layer DINO field")
        tokens = self.value(evidence_layers.mean(dim=1))
        attention = self._attention(self.explicit_queries, tokens)
        explicit = torch.einsum("ben,bnd->bed", attention, tokens)
        latent_attention = self._attention(self.latent_queries, tokens)
        latent = torch.einsum("bln,bnd->bld", latent_attention, tokens)
        coords = self._coordinates(tokens.device, tokens.dtype)
        center = torch.einsum("ben,nd->bed", attention, coords)
        offsets = 0.10 * torch.tanh(self.part_offsets).unsqueeze(0)
        part_coordinates = (center.unsqueeze(2) + offsets).clamp(0.0, 1.0)
        part_scales = 0.03 + 0.27 * torch.sigmoid(self.part_scales_raw).unsqueeze(0)
        yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, self.grid_hw[0], device=tokens.device, dtype=tokens.dtype), torch.linspace(0.0, 1.0, self.grid_hw[1], device=tokens.device, dtype=tokens.dtype), indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).view(1, 1, 1, self.grid_hw[0], self.grid_hw[1], 2)
        distance = ((grid - part_coordinates.unsqueeze(-2).unsqueeze(-2)) / part_scales.unsqueeze(-2).unsqueeze(-2).clamp_min(1e-4)).square().sum(-1)
        soft_masks = torch.exp(-0.5 * distance).amax(dim=2)
        presence_logits = self.presence(explicit).squeeze(-1)
        observability_logits = self.observability(explicit).squeeze(-1)
        max_state = max(head.out_features for head in self.state_heads)
        state_logits = explicit.new_zeros(batch, self.num_explicit, max_state)
        for index, head in enumerate(self.state_heads):
            state_logits[:, index, : head.out_features] = head(explicit[:, index])
        pos = torch.nn.functional.normalize(self.positive_prototypes, dim=-1)
        neg = torch.nn.functional.normalize(self.negative_prototypes, dim=-1)
        normalized = torch.nn.functional.normalize(explicit, dim=-1)
        margin = (normalized * pos.unsqueeze(0)).sum(-1) - (normalized * neg.unsqueeze(0)).sum(-1)
        reliability = torch.sigmoid(observability_logits) * torch.sigmoid(margin / self.reliability_tau) * self.view_consistency_ema.view(1, -1)
        derived = self._derived_atoms(presence_logits, state_logits)
        return {
            "explicit_tokens": explicit,
            "latent_tokens": latent,
            "presence_logits": presence_logits,
            "observability_logits": observability_logits,
            "state_logits": state_logits,
            "type_logits_actor": state_logits[:, 2:5, :4],
            "part_coordinates": part_coordinates,
            "part_scales": part_scales,
            "soft_masks": soft_masks,
            "derived_atom_probs": derived,
            "reliability": reliability,
            "field_attention": attention,
            "latent_attention": latent_attention,
            "prototype_margin": margin,
        }
