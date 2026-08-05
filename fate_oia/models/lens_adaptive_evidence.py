from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class LENSAdaptiveEvidence(nn.Module):
    """Reason-specific, null-aware evidence pooling without factor-by-token-by-channel materialization."""

    def __init__(self, dim: int = 384, reason_dim: int = 21, layer_count: int = 3, tau_min: float = 0.35, tau_max: float = 2.0) -> None:
        super().__init__()
        self.reason_dim, self.layer_count = reason_dim, layer_count
        self.tau_min, self.tau_max = tau_min, tau_max
        self.layer_router = nn.Parameter(torch.zeros(reason_dim, layer_count))
        self.layer_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(layer_count))
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.temperature_mlp = nn.Sequential(nn.Linear(dim + 2, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
        self.null_mlp = nn.Sequential(nn.Linear(dim + 2, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def forward(self, reason_nodes_source: Tensor, patch_tokens_by_layer: Tensor, soft_region_prior: Tensor | None = None) -> dict[str, Tensor]:
        b, layers, n, d = patch_tokens_by_layer.shape
        if layers != self.layer_count:
            raise ValueError(f"expected {self.layer_count} layers, got {layers}")
        layer_weight = torch.softmax(self.layer_router, dim=-1)
        projected = torch.stack([self.layer_proj[idx](patch_tokens_by_layer[:, idx]) for idx in range(layers)], dim=1)
        keys = self.key_proj(projected)
        values = self.value_proj(projected)
        query = self.query_proj(reason_nodes_source)
        score_by_layer = torch.einsum("brd,bsnd->brsn", query, keys) / math.sqrt(d)
        score = torch.einsum("rs,brsn->brn", layer_weight, score_by_layer)
        if soft_region_prior is not None:
            if soft_region_prior.shape != (b, self.reason_dim, n):
                raise ValueError("soft_region_prior must be [B,21,N]")
            score = score + soft_region_prior.clamp(-2.0, 2.0)
        mean = score.mean(-1)
        std = score.std(-1, unbiased=False).clamp_min(1e-6)
        topk = score.topk(min(32, n), dim=-1).values.mean(-1)
        snr = (topk - mean) / std
        temperature_input = torch.cat([reason_nodes_source, snr.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
        temperature = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(self.temperature_mlp(temperature_input).squeeze(-1))
        null_logit = self.null_mlp(temperature_input).squeeze(-1)
        augmented = torch.cat([score / temperature.unsqueeze(-1), null_logit.unsqueeze(-1)], dim=-1)
        mass = torch.softmax(augmented, dim=-1)
        evidence_map, null_mass = mass[..., :-1], mass[..., -1]
        # Pool per layer first: [B,R,S,D], then mix layers. No [B,R,N,D] tensor is formed.
        pooled_per_layer = torch.einsum("brn,bsnd->brsd", evidence_map, values)
        evidence_token = torch.einsum("rs,brsd->brd", layer_weight, pooled_per_layer)
        evidence_token = evidence_token / (1.0 - null_mass).clamp_min(1e-6).unsqueeze(-1)
        entropy = -(mass.clamp_min(1e-9) * mass.clamp_min(1e-9).log()).sum(-1)
        return {
            "evidence_map": evidence_map,
            "evidence_null_mass": null_mass,
            "evidence_token": evidence_token,
            "evidence_temperature": temperature,
            "evidence_snr": snr,
            "evidence_entropy": entropy,
            "evidence_layer_weight": layer_weight,
            "evidence_score_mean": mean,
            "evidence_score_std": std,
            "evidence_topk_gap": topk - mean,
        }
