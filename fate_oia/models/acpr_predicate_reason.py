from __future__ import annotations

import torch
from torch import nn


class ACPRPredicateReasoner(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21, num_predicates: int = 32) -> None:
        super().__init__()
        self.reason_dim = reason_dim
        self.num_predicates = num_predicates
        self.proj = nn.Sequential(nn.Linear(num_predicates + dim, dim), nn.GELU(), nn.Linear(dim, reason_dim))
        self.gate = nn.Parameter(torch.full((reason_dim,), -2.944))  # sigmoid ~= 0.05

    def forward(self, reason_nodes: torch.Tensor, predicate_probs: torch.Tensor, predicate_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        context = predicate_tokens.mean(1)
        raw = self.proj(torch.cat([predicate_probs, context], dim=-1))
        bounded = raw.clamp(-1.0, 1.0) * 0.20
        gate = torch.sigmoid(self.gate).clamp(max=0.20).view(1, -1)
        delta = (bounded * gate).clamp(-0.20, 0.20)
        return {"predicate_reason_delta": delta, "predicate_reason_gate": gate.expand_as(delta)}
