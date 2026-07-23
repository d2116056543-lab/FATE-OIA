from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PRECISEVisualRereader(nn.Module):
    def __init__(self, dim: int = 384, sampling_points_per_layer: int = 4, gamma_init: float = 0.08, gamma_max: float = 0.35) -> None:
        super().__init__()
        self.points = sampling_points_per_layer
        self.gamma_max = gamma_max
        init = max(min(gamma_init / gamma_max, 1.0 - 1e-4), 1e-4)
        self.gamma_raw = nn.Parameter(torch.tensor(math.log(init / (1.0 - init))))
        self.action_field_query = nn.Linear(dim, dim, bias=False)
        self.reason_field_query = nn.Linear(dim, dim, bias=False)
        self.error_mlp = nn.Sequential(nn.Linear(dim + 6, dim), nn.GELU(), nn.Linear(dim, 2))
        self.action_projection = nn.Linear(dim, dim)
        self.reason_projection = nn.Linear(dim, dim)

    def _field_attention(self, tokens: torch.Tensor, fields: torch.Tensor, query: nn.Linear) -> torch.Tensor:
        logits = torch.einsum("bcd,bed->bce", query(tokens), fields) / (tokens.shape[-1] ** 0.5)
        top = logits.topk(k=min(2, logits.shape[-1]), dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, top, True)
        return torch.softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)

    @staticmethod
    def _demand_state(logits: torch.Tensor, attention: torch.Tensor, reliability: torch.Tensor) -> torch.Tensor:
        probability = torch.sigmoid(logits)
        entropy = -(probability.clamp_min(1e-8) * probability.clamp_min(1e-8).log() + (1.0 - probability).clamp_min(1e-8) * (1.0 - probability).clamp_min(1e-8).log())
        support = torch.einsum("bce,be->bc", attention, reliability)
        veto = torch.einsum("bce,be->bc", attention, 1.0 - reliability)
        conflict = (support - veto).abs()
        coverage = (attention > 0).float().sum(-1) / attention.shape[-1]
        return torch.stack([probability, entropy, support, veto, conflict, coverage], dim=-1)

    def _references(self, tokens: torch.Tensor, attention: torch.Tensor, part_coordinates: torch.Tensor, part_valid: torch.Tensor, demand_state: torch.Tensor, layer_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid = part_valid.to(part_coordinates).view(1, *part_valid.shape, 1)
        field_center = (part_coordinates * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        base = torch.einsum("bce,bed->bcd", attention, field_center)
        error = 0.10 * torch.tanh(self.error_mlp(torch.cat([tokens, demand_state], dim=-1)))
        reference = base + error
        offsets = torch.tensor([[-0.03, -0.03], [0.03, -0.03], [-0.03, 0.03], [0.03, 0.03]], device=tokens.device, dtype=tokens.dtype)[: self.points]
        raw_references = reference[:, :, None, None, :] + offsets.view(1, 1, 1, self.points, 2)
        references = raw_references.clamp(0.0, 1.0)
        entropy = -(attention.clamp_min(1e-8) * attention.clamp_min(1e-8).log()).sum(-1)
        return references.expand(-1, -1, layer_count, -1, -1), raw_references.expand(-1, -1, layer_count, -1, -1), entropy

    def _sample(self, layers: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        batch, layers_count, _, dim = layers.shape
        height, width = 45, 80
        sampled_layers = []
        for layer in range(layers_count):
            image = layers[:, layer].transpose(1, 2).reshape(batch, dim, height, width)
            grid = (references[:, :, layer].reshape(batch, -1, 1, 2) * 2.0 - 1.0).to(dtype=image.dtype)
            sampled = F.grid_sample(image, grid, mode="bilinear", align_corners=True).squeeze(-1).transpose(1, 2)
            sampled_layers.append(sampled.reshape(batch, references.shape[1], self.points, dim))
        return torch.stack(sampled_layers, dim=2).mean(dim=(2, 3))

    def forward(self, action_tokens: torch.Tensor, reason_tokens: torch.Tensor, evidence: dict[str, torch.Tensor], action_layers: torch.Tensor, reason_layers: torch.Tensor, action_logits: torch.Tensor, reason_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        fields = evidence["explicit_tokens"].detach()
        coordinates = evidence["part_coordinates"].detach()
        part_valid = evidence["part_valid"].detach()
        reliability = evidence["reliability"].detach()
        action_attention = self._field_attention(action_tokens, fields, self.action_field_query)
        reason_attention = self._field_attention(reason_tokens, fields, self.reason_field_query)
        action_demand = self._demand_state(action_logits, action_attention, reliability)
        reason_demand = self._demand_state(reason_logits, reason_attention, reliability)
        joined_tokens = torch.cat([action_tokens, reason_tokens], dim=1)
        joined_attention = torch.cat([action_attention, reason_attention], dim=1)
        demand = torch.cat([action_demand, reason_demand], dim=1)
        references, raw_references, entropy = self._references(joined_tokens, joined_attention, coordinates, part_valid, demand, action_layers.shape[1])
        action_samples = self._sample(action_layers, references[:, :4])
        reason_samples = self._sample(reason_layers, references[:, 4:])
        gamma = self.gamma_max * torch.sigmoid(self.gamma_raw)
        action_delta = gamma * self.action_projection(action_samples)
        reason_delta = gamma * self.reason_projection(reason_samples)
        points = references
        variance = points.var(dim=(0, 1, 2, 3), unbiased=False)
        center_collapse = ((points - 0.5).abs().amax(dim=-1) < 0.10).float().mean()
        out_of_bounds = ((raw_references < 0.0) | (raw_references > 1.0)).any(dim=-1).float().mean()
        return {
            "action_reread_delta": action_delta,
            "reason_reread_delta": reason_delta,
            "reference_points": points,
            "raw_reference_points": raw_references,
            "sampling_weights": joined_attention,
            "reference_entropy": entropy.squeeze(-1),
            "evidence_demand_state": demand,
            "reference_point_variance": variance,
            "center_collapse_rate": center_collapse,
            "out_of_bounds_rate": out_of_bounds,
        }
