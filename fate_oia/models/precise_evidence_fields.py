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
        self.explicit_part_queries = nn.Parameter(torch.randn(self.num_explicit, self.max_parts, dim) * 0.02)
        self.latent_part_queries = nn.Parameter(torch.randn(latent_slots, latent_parts, dim) * 0.02)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.latent_key = nn.Linear(dim, dim, bias=False)
        self.latent_value = nn.Linear(dim, dim, bias=False)
        self.presence = nn.Linear(dim, 1)
        self.observability = nn.Linear(dim, 1)
        state_widths = [max(1, len(field.get("state_schema", field.get("type_schema", [])))) for field in fields]
        self.state_heads = nn.ModuleList([nn.Linear(dim, width) for width in state_widths])
        self.max_state_width = max(state_widths)
        state_channel_valid = torch.arange(self.max_state_width).view(1, -1) < torch.tensor(state_widths).view(-1, 1)
        self.register_buffer("state_channel_valid", state_channel_valid)
        self.actor_part_type = nn.Linear(dim, 4)
        self.actor_part_occupancy = nn.Linear(dim, 1)
        self.part_offsets = nn.Parameter(torch.zeros(self.num_explicit, self.max_parts, 2))
        self.part_scales_raw = nn.Parameter(torch.zeros(self.num_explicit, self.max_parts, 2))
        self.positive_prototypes = nn.Parameter(torch.randn(self.num_explicit, dim) * 0.02)
        self.negative_prototypes = nn.Parameter(torch.randn(self.num_explicit, dim) * 0.02)
        self.register_buffer("view_consistency_ema", torch.full((self.num_explicit,), 0.75))
        self.register_buffer("part_count", torch.tensor([int(field["num_parts"]) for field in fields], dtype=torch.long))
        part_valid = torch.arange(self.max_parts).view(1, -1) < self.part_count.view(-1, 1)
        self.register_buffer("part_valid", part_valid)
        geometry_codes = {"points": 0, "region": 1, "ordered_curve": 2}
        self.register_buffer("geometry_type", torch.tensor([geometry_codes[field["part_type"]] for field in fields], dtype=torch.long))
        curve_anchor = torch.full((self.num_explicit, self.max_parts), -1.0)
        for index, field in enumerate(fields):
            if field["part_type"] == "ordered_curve":
                curve_anchor[index, : int(field["num_parts"])] = torch.linspace(0.15, 0.95, int(field["num_parts"]))
        self.register_buffer("curve_y_anchor", curve_anchor)
        names = [field["name"] for field in fields]
        actor_names = ("actor_left", "actor_center", "actor_right")
        if any(name not in names for name in actor_names):
            raise ValueError("PRECISE requires left/center/right actor evidence fields")
        actor_indices = [names.index(name) for name in actor_names]
        self.register_buffer("actor_indices", torch.tensor(actor_indices, dtype=torch.long))
        mirror_names = {
            "actor_left": "actor_right", "actor_right": "actor_left",
            "drivable_left": "drivable_right", "drivable_right": "drivable_left",
            "boundary_left": "boundary_right", "boundary_right": "boundary_left",
        }
        self.register_buffer("mirror_field_indices", torch.tensor([names.index(mirror_names.get(name, name)) for name in names], dtype=torch.long))

    @torch.no_grad()
    def update_view_consistency(self, canonical: torch.Tensor, mirrored: torch.Tensor, momentum: float = 0.95) -> torch.Tensor:
        if canonical.shape != mirrored.shape or canonical.shape[1] != self.num_explicit:
            raise ValueError("View-consistency evidence tensors do not match the explicit schema")
        aligned = mirrored[:, self.mirror_field_indices]
        score = torch.nn.functional.cosine_similarity(canonical, aligned, dim=-1).clamp(0.0, 1.0).mean(0)
        self.view_consistency_ema.mul_(momentum).add_(score, alpha=1.0 - momentum)
        return score

    def _coordinates(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        height, width = self.grid_hw
        yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, height, device=device, dtype=dtype), torch.linspace(0.0, 1.0, width, device=device, dtype=dtype), indexing="ij")
        return torch.stack([xx, yy], dim=-1).reshape(height * width, 2)

    def _part_attention(self, queries: torch.Tensor, tokens: torch.Tensor, coordinates: torch.Tensor, curve_anchor: torch.Tensor | None = None, key: nn.Linear | None = None) -> torch.Tensor:
        projected = self.key(tokens) if key is None else key(tokens)
        logits = torch.einsum("epd,bnd->bepn", queries, projected) / (tokens.shape[-1] ** 0.5)
        if curve_anchor is not None:
            anchors = curve_anchor.clamp_min(0.0)
            y = coordinates[:, 1].view(1, 1, 1, -1)
            spatial = -40.0 * (y - anchors.view(1, *anchors.shape, 1)).square()
            logits = logits + spatial * (curve_anchor >= 0).view(1, *curve_anchor.shape, 1)
        return torch.softmax(logits, dim=-1)

    def _derived_atoms(self, presence: torch.Tensor, state_logits: torch.Tensor, actor_type_probability: torch.Tensor) -> dict[str, torch.Tensor]:
        p = torch.sigmoid(presence)
        state = torch.sigmoid(state_logits)
        name_to_idx = {field["name"]: idx for idx, field in enumerate(self.fields)}
        zero = p.new_zeros(p.shape[0])
        def probability(name: str) -> torch.Tensor:
            return p[:, name_to_idx[name]] if name in name_to_idx else zero
        def typed(name: str, index: int) -> torch.Tensor:
            return probability(name) * state[:, name_to_idx[name], index] if name in name_to_idx else zero
        atom = {
            "traffic_light_visible": probability("traffic_light"),
            "traffic_light_red": typed("traffic_light", 0),
            "traffic_light_green": typed("traffic_light", 1),
            "traffic_sign_visible": probability("traffic_sign"),
            "front_vehicle_visible": actor_type_probability[:, 1, 0],
            "front_pedestrian_visible": actor_type_probability[:, 1, 1],
            "front_rider_visible": actor_type_probability[:, 1, 2],
            "front_other_obstacle": actor_type_probability[:, 1, 3],
            "left_occupied": probability("actor_left"),
            "center_occupied": probability("actor_center"),
            "right_occupied": probability("actor_right"),
            "left_drivable": probability("drivable_left"),
            "center_drivable": probability("drivable_center"),
            "right_drivable": probability("drivable_right"),
            "left_boundary_visible": probability("boundary_left"),
            "right_boundary_visible": probability("boundary_right"),
            "left_solid_boundary": typed("boundary_left", 0),
            "right_solid_boundary": typed("boundary_right", 0),
        }
        return atom

    def _certificate_probability(self, atoms: dict[str, torch.Tensor]) -> torch.Tensor:
        names = {
            "traffic_light": "traffic_light_visible", "traffic_sign": "traffic_sign_visible",
            "actor_left": "left_occupied", "actor_center": "center_occupied", "actor_right": "right_occupied",
            "drivable_left": "left_drivable", "drivable_center": "center_drivable", "drivable_right": "right_drivable",
            "boundary_left": "left_boundary_visible", "boundary_right": "right_boundary_visible",
        }
        return torch.stack([atoms[names[field["name"]]] for field in self.fields], dim=1)

    def forward(self, evidence_layers: torch.Tensor, latent_layers: torch.Tensor | None = None) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        batch, layer_count, token_count, dim = evidence_layers.shape
        if (layer_count, token_count, dim) != (3, 3600, 384):
            raise ValueError("PRECISE evidence receives the uncompressed three-layer DINO field")
        tokens = self.value(evidence_layers.mean(dim=1))
        coords = self._coordinates(tokens.device, tokens.dtype)
        part_queries = self.explicit_queries.unsqueeze(1) + self.explicit_part_queries
        part_attention = self._part_attention(part_queries, tokens, coords, self.curve_y_anchor)
        part_attention = part_attention * self.part_valid.view(1, self.num_explicit, self.max_parts, 1)
        part_features = torch.einsum("bepn,bnd->bepd", part_attention, tokens)
        explicit = part_features.sum(2) / self.part_count.to(tokens).view(1, -1, 1)
        latent_input = evidence_layers.detach() if latent_layers is None else latent_layers
        if latent_input.shape != evidence_layers.shape:
            raise ValueError("PRECISE latent visual field must match the explicit field geometry")
        latent_source = self.latent_value(latent_input.mean(dim=1))
        latent_attention = self._part_attention(self.latent_part_queries, latent_source, coords, key=self.latent_key)
        latent_parts = torch.einsum("blpn,bnd->blpd", latent_attention, latent_source)
        latent = latent_parts.mean(2)
        center = torch.einsum("bepn,nd->bepd", part_attention, coords)
        offsets = 0.10 * torch.tanh(self.part_offsets).unsqueeze(0)
        part_coordinates = (center + offsets).clamp(0.0, 1.0)
        curve_mask = (self.curve_y_anchor >= 0).view(1, self.num_explicit, self.max_parts)
        anchored_y = self.curve_y_anchor.clamp_min(0.0).view(1, self.num_explicit, self.max_parts)
        part_coordinates = torch.stack([part_coordinates[..., 0], torch.where(curve_mask, anchored_y, part_coordinates[..., 1])], dim=-1)
        part_scales = (0.03 + 0.27 * torch.sigmoid(self.part_scales_raw)).unsqueeze(0).expand(batch, -1, -1, -1)
        yy, xx = torch.meshgrid(torch.linspace(0.0, 1.0, self.grid_hw[0], device=tokens.device, dtype=tokens.dtype), torch.linspace(0.0, 1.0, self.grid_hw[1], device=tokens.device, dtype=tokens.dtype), indexing="ij")
        grid = torch.stack([xx, yy], dim=-1).view(1, 1, 1, self.grid_hw[0], self.grid_hw[1], 2)
        distance = ((grid - part_coordinates.unsqueeze(-2).unsqueeze(-2)) / part_scales.unsqueeze(-2).unsqueeze(-2).clamp_min(1e-4)).square().sum(-1)
        soft_masks = torch.exp(-0.5 * distance).amax(dim=2)
        presence_logits = self.presence(explicit).squeeze(-1)
        observability_logits = self.observability(explicit).squeeze(-1)
        state_logits = explicit.new_zeros(batch, self.num_explicit, self.max_state_width)
        for index, head in enumerate(self.state_heads):
            state_logits[:, index, : head.out_features] = head(explicit[:, index])
        actor_part_type_logits = self.actor_part_type(part_features[:, self.actor_indices])
        actor_part_occupancy_logits = self.actor_part_occupancy(part_features[:, self.actor_indices]).squeeze(-1)
        actor_valid = self.part_valid[self.actor_indices].view(1, 3, self.max_parts, 1)
        actor_occupancy_probability = torch.sigmoid(actor_part_occupancy_logits) * actor_valid.squeeze(-1).to(actor_part_occupancy_logits)
        actor_presence_probability = 1.0 - (1.0 - actor_occupancy_probability).prod(dim=2)
        actor_joint_probability = actor_occupancy_probability.unsqueeze(-1) * torch.sigmoid(actor_part_type_logits)
        actor_type_probability = 1.0 - (1.0 - actor_joint_probability).prod(dim=2)
        actor_noisy_or_logits = torch.logit(actor_type_probability.clamp(1e-6, 1.0 - 1e-6))
        presence_logits = presence_logits.clone()
        presence_logits[:, self.actor_indices] = torch.logit(actor_presence_probability.clamp(1e-6, 1.0 - 1e-6)).to(presence_logits)
        state_logits = state_logits.clone()
        state_logits[:, self.actor_indices, :4] = actor_noisy_or_logits.to(state_logits)
        pos = torch.nn.functional.normalize(self.positive_prototypes, dim=-1)
        neg = torch.nn.functional.normalize(self.negative_prototypes, dim=-1)
        normalized = torch.nn.functional.normalize(explicit, dim=-1)
        margin = (normalized * pos.unsqueeze(0)).sum(-1) - (normalized * neg.unsqueeze(0)).sum(-1)
        derived = self._derived_atoms(presence_logits, state_logits, actor_type_probability)
        certificate_probability = self._certificate_probability(derived)
        consistency_snapshot = self.view_consistency_ema.detach().clone().view(1, -1)
        reliability = certificate_probability * torch.sigmoid(observability_logits) * torch.sigmoid(margin / self.reliability_tau) * consistency_snapshot
        field_attention = part_attention.sum(2) / self.part_count.to(tokens).view(1, -1, 1)
        return {
            "explicit_tokens": explicit,
            "latent_tokens": latent,
            "presence_logits": presence_logits,
            "observability_logits": observability_logits,
            "state_logits": state_logits,
            "state_channel_valid": self.state_channel_valid,
            "type_logits_actor": state_logits[:, self.actor_indices, :4],
            "actor_part_type_logits": actor_part_type_logits[:, :, :4],
            "actor_part_occupancy_logits": actor_part_occupancy_logits[:, :, :4],
            "actor_type_probability": actor_type_probability,
            "part_coordinates": part_coordinates,
            "part_scales": part_scales,
            "soft_masks": soft_masks,
            "derived_atom_probs": derived,
            "certificate_probability": certificate_probability,
            "reliability": reliability,
            "field_attention": field_attention,
            "explicit_part_attention": part_attention,
            "latent_attention": latent_attention,
            "latent_part_attention": latent_attention,
            "explicit_part_features": part_features,
            "latent_part_features": latent_parts,
            "prototype_margin": margin,
            "part_valid": self.part_valid,
            "geometry_type": self.geometry_type,
            "source_tokens": tokens,
        }

    def latent_parameters(self) -> list[nn.Parameter]:
        return [self.latent_part_queries, *self.latent_key.parameters(), *self.latent_value.parameters()]

    def explicit_parameters(self) -> list[nn.Parameter]:
        latent = {id(parameter) for parameter in self.latent_parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in latent]
