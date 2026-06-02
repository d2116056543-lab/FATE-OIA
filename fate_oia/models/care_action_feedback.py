from __future__ import annotations

import torch
from torch import nn


class ReasonToActionSafeFeedback(nn.Module):
    def __init__(self, action_dim: int = 4, reason_dim: int = 21, cap: float = 0.04, warmup_epochs: int = 2) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.cap = cap
        self.warmup_epochs = warmup_epochs
        self.proj = nn.Sequential(nn.Linear(reason_dim * 2 + action_dim, 64), nn.GELU(), nn.Linear(64, action_dim))
        self.gate = nn.Sequential(nn.Linear(reason_dim + action_dim, 64), nn.GELU(), nn.Linear(64, action_dim), nn.Sigmoid())

    def forward(self, base_action: torch.Tensor, reason_delta: torch.Tensor, reason_reliability: torch.Tensor, epoch: int = 0, force_shutdown: bool = False) -> dict[str, torch.Tensor]:
        raw = self.proj(torch.cat([reason_delta, reason_reliability, base_action], dim=-1))
        gate = self.gate(torch.cat([reason_reliability, base_action], dim=-1))
        active = (epoch >= self.warmup_epochs) and not force_shutdown
        action_delta = torch.tanh(raw) * self.cap * gate
        if not active:
            action_delta = action_delta * 0.0
        candidate = base_action + action_delta
        state = "candidate" if active else "base_fallback"
        return {
            "action_delta": action_delta,
            "action_gate": gate,
            "action_final_candidate_logits": candidate,
            "action_logits": candidate if active else base_action,
            "action_safe_state": state,
            "action_residual_abs_max": action_delta.detach().abs().max(),
        }
