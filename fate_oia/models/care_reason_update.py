from __future__ import annotations

import torch
from torch import nn


class EvidenceToReasonUpdate(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21, cap: float = 0.30, tail_cap: float = 0.45) -> None:
        super().__init__()
        self.reason_dim = reason_dim
        self.cap = cap
        self.tail_cap = tail_cap
        self.delta = nn.Sequential(nn.Linear(dim * 2 + 1, dim), nn.GELU(), nn.Linear(dim, 1))
        self.reliability = nn.Sequential(nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.register_buffer("tail_mask", torch.zeros(reason_dim))
        self.tail_mask[[5, 6, 9, 11, 12, 14]] = 1.0

    def forward(self, base_reason: torch.Tensor, reason_tokens: torch.Tensor, evidence_tokens: torch.Tensor, active_mask: torch.Tensor, route_reliability: torch.Tensor) -> dict[str, torch.Tensor]:
        rel = self.reliability(torch.cat([reason_tokens, evidence_tokens], dim=-1)).squeeze(-1) * route_reliability
        raw_delta = self.delta(torch.cat([reason_tokens, evidence_tokens, rel.unsqueeze(-1)], dim=-1)).squeeze(-1)
        caps = self.cap + (self.tail_cap - self.cap) * self.tail_mask.to(base_reason.device).view(1, -1)
        delta = torch.tanh(raw_delta) * caps * rel
        delta = delta.masked_fill(~active_mask, 0.0)
        return {"reason_delta": delta, "reason_reliability": rel, "reason_logits": base_reason + delta, "evidence_backed_reason_mask": (rel > 0.15) & active_mask}
