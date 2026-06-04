from __future__ import annotations

import torch
from torch import nn


class ControlledCrossExpertExchange(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, heads: int = 4, a2r_max_scale: float = 0.75, r2a_max_scale: float = 0.25) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.a2r_max_scale = a2r_max_scale
        self.r2a_max_scale = r2a_max_scale
        self.a2r = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.r2a = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.a2r_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1), nn.Sigmoid())
        self.r2a_gate = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1), nn.Sigmoid())
        self.norm_a = nn.LayerNorm(dim)
        self.norm_r = nn.LayerNorm(dim)

    def forward(self, action_tokens: torch.Tensor, reason_tokens: torch.Tensor, q_ar: torch.Tensor | None = None, readiness: dict | None = None) -> dict[str, torch.Tensor | dict[str, float]]:
        readiness = readiness or {}
        action_detached = action_tokens.detach()
        msg_r, _ = self.a2r(self.norm_r(reason_tokens), self.norm_a(action_detached), self.norm_a(action_detached), need_weights=False)
        a2r_gate = self.a2r_gate(reason_tokens)
        reason_out = reason_tokens + self.a2r_max_scale * a2r_gate * msg_r
        r2a_active = bool(readiness.get("r2a_active", False))
        if q_ar is None:
            rel = action_tokens.new_zeros(action_tokens.shape[0], self.action_dim, 1)
        else:
            rel = q_ar.detach().mean(dim=2, keepdim=True).clamp(0.0, 1.0)
        if r2a_active:
            msg_a, _ = self.r2a(self.norm_a(action_tokens), self.norm_r(reason_out.detach()), self.norm_r(reason_out.detach()), need_weights=False)
            gate = torch.sigmoid(self.r2a_gate(action_tokens).mean(dim=-1, keepdim=True)) * rel
            action_out = action_tokens + self.r2a_max_scale * gate * torch.tanh(msg_a)
        else:
            gate = action_tokens.new_zeros(action_tokens.shape[0], self.action_dim, 1)
            action_out = action_tokens
        stats = {
            "a2r_gate_mean": float(a2r_gate.detach().mean().cpu()),
            "r2a_gate_mean": float(gate.detach().mean().cpu()),
            "r2a_active_rate": float(1.0 if r2a_active else 0.0),
            "action_token_delta_norm": float((action_out - action_tokens).detach().norm(dim=-1).mean().cpu()),
            "reason_token_delta_norm": float((reason_out - reason_tokens).detach().norm(dim=-1).mean().cpu()),
        }
        return {"action_tokens": action_out, "reason_tokens": reason_out, "stats": stats}
