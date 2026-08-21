from __future__ import annotations

import math

import torch
from torch import nn


class TIDAReasonReader(nn.Module):
    """Private reason correction with an explicit detach firewall."""

    def __init__(self, dim: int = 384, num_reasons: int = 21, kappa: float = 0.12) -> None:
        super().__init__()
        self.num_reasons = int(num_reasons)
        self.kappa = float(kappa)
        self.reason_query = nn.Linear(dim, dim)
        self.factor_key = nn.Linear(dim, dim)
        self.factor_value = nn.Linear(dim, dim)
        self.private_attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.delta_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(
        self,
        reason_nodes: torch.Tensor,
        predicate_state: torch.Tensor,
        action_innovation: torch.Tensor,
        reliability: torch.Tensor,
        *,
        temporal_scale: float | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        factors = torch.cat([predicate_state.detach(), action_innovation.detach()], dim=1)
        weights = reliability.detach().clamp(0, 1)
        query = self.reason_query(reason_nodes)
        keys = self.factor_key(factors)
        values = self.factor_value(factors)
        score = torch.einsum("brd,bfd->brf", query, keys) / math.sqrt(keys.shape[-1])
        score = score + weights.clamp_min(1e-7).log()[:, None]
        attention = torch.softmax(score, dim=-1) * weights[:, None]
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-7)
        private = torch.einsum("brf,bfd->brd", attention, values)
        private = private + self.private_attention(private, private, private, need_weights=False)[0]
        raw_delta = self.delta_head(private + reason_nodes).squeeze(-1)
        scale = torch.as_tensor(temporal_scale, device=raw_delta.device, dtype=raw_delta.dtype)
        delta = scale * self.kappa * torch.tanh(raw_delta / self.kappa)
        no_reliable_factor = ~weights.gt(0).any(dim=-1)
        delta = torch.where(no_reliable_factor[:, None], torch.zeros_like(delta), delta)
        return {
            "reason_temporal_attention": attention,
            "reason_private_token": private,
            "reason_raw_temporal_delta": raw_delta,
            "reason_temporal_delta": delta,
        }
