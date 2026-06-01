from __future__ import annotations

import torch
from torch import nn


class SUREBoundedResidualRefiner(nn.Module):
    def __init__(self, dim: int, action_dim: int = 4, reason_dim: int = 21, action_cap: float = 0.35, reason_cap: float = 0.75) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.action_cap = float(action_cap)
        self.reason_cap = float(reason_cap)
        self.norm = nn.LayerNorm(dim)
        self.action_delta = nn.Linear(dim, action_dim)
        self.reason_delta = nn.Linear(dim, reason_dim)

    def forward(self, label_tokens: torch.Tensor, action_base: torch.Tensor, reason_base: torch.Tensor) -> dict[str, torch.Tensor | dict[str, float]]:
        action_context = self.norm(label_tokens[:, : self.action_dim].mean(1))
        reason_context = self.norm(label_tokens[:, self.action_dim :].mean(1))
        raw_action = self.action_delta(action_context)
        raw_reason = self.reason_delta(reason_context)
        action_delta = torch.tanh(raw_action) * self.action_cap
        reason_delta = torch.tanh(raw_reason) * self.reason_cap
        action_final = action_base + action_delta
        reason_final = reason_base + reason_delta
        stats = {
            "action_residual_abs_mean": float(action_delta.detach().abs().mean().cpu().item()),
            "reason_residual_abs_mean": float(reason_delta.detach().abs().mean().cpu().item()),
            "action_cap": self.action_cap,
            "reason_cap": self.reason_cap,
        }
        return {
            "action_logits": action_final,
            "reason_logits": reason_final,
            "action_delta": action_delta,
            "reason_delta": reason_delta,
            "action_safe_stats": stats,
        }
