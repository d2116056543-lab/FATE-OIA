from __future__ import annotations

import torch
from torch import nn


class EvidenceGatedReasonResidual(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21, tail_labels: tuple[int, ...] = (12, 9, 5, 14, 6, 11, 10, 13), common_scale: float = 0.14, tail_scale: float = 0.22) -> None:
        super().__init__()
        self.reason_dim = reason_dim
        self.tail_labels = tuple(int(x) for x in tail_labels)
        self.delta = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.gate = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1), nn.Sigmoid())
        self.common_scale = float(common_scale)
        self.tail_scale = float(tail_scale)

    def forward(self, reason_tokens: torch.Tensor, base_reason_logits: torch.Tensor, causal_reason_logits: torch.Tensor, evidence: dict) -> dict[str, torch.Tensor]:
        ev = evidence["evidence_tokens"]
        mask = evidence["evidence_mask"].to(ev.dtype).unsqueeze(-1)
        ev_summary = (ev * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        ev_expand = ev_summary.unsqueeze(1).expand(-1, reason_tokens.shape[1], -1)
        x = torch.cat([reason_tokens, ev_expand], dim=-1)
        raw_delta = self.delta(x).squeeze(-1)
        gate = self.gate(x).squeeze(-1)
        quality = evidence.get("reason_quality")
        if isinstance(quality, torch.Tensor):
            gate = gate * quality[:, : gate.shape[1]].to(gate.device).to(gate.dtype)
        scales = raw_delta.new_full((self.reason_dim,), self.common_scale)
        if self.tail_labels:
            idx = torch.tensor([i for i in self.tail_labels if 0 <= i < self.reason_dim], device=raw_delta.device)
            if idx.numel():
                scales[idx] = self.tail_scale
        reason_delta = torch.tanh(raw_delta) * scales.view(1, -1)
        return {"reason_delta": reason_delta, "reason_gate": gate}


class EvidenceGatedActionResidual(nn.Module):
    def __init__(self, reason_dim: int = 21, action_dim: int = 4, beta_max: float = 0.06) -> None:
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(reason_dim * 2 + 1, 64), nn.GELU(), nn.Linear(64, action_dim))
        self.beta = nn.Parameter(torch.tensor(0.0))
        self.beta_max = float(beta_max)

    def forward(self, base_action_logits: torch.Tensor, action_reason_logits: torch.Tensor, evidence: dict, reason_logits: torch.Tensor, base_reason_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        q = evidence.get("evidence_quality")
        if isinstance(q, torch.Tensor):
            quality = q.mean(1, keepdim=True).to(reason_logits.device).to(reason_logits.dtype)
        else:
            quality = reason_logits.new_zeros((reason_logits.shape[0], 1))
        x = torch.cat([reason_logits, reason_logits - base_reason_logits, quality], dim=-1)
        beta = torch.clamp(self.beta, 0.0, self.beta_max)
        delta = torch.tanh(self.proj(x)) * beta * quality
        return {"action_delta": delta, "action_beta": beta.expand_as(delta)}

