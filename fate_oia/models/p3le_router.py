from __future__ import annotations

import torch
from torch import nn


class ParetoSafeRouter(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, action_residual_cap: float = 0.04) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.action_residual_cap = float(action_residual_cap)
        self.action_router = nn.Sequential(nn.Linear(action_dim * 3 + dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, action_dim), nn.Sigmoid())
        self.reason_router = nn.Sequential(nn.Linear(reason_dim * 4 + dim, dim), nn.GELU(), nn.Linear(dim, reason_dim), nn.Sigmoid())
        self.reason_calibration = nn.Linear(reason_dim, reason_dim)

    def forward(
        self,
        a_action: torch.Tensor,
        r_reason: torch.Tensor,
        pair_action: torch.Tensor,
        pair_reason: torch.Tensor,
        action_set_logits: torch.Tensor,
        reliability: torch.Tensor,
        shared_context: torch.Tensor,
        router_scale: float,
    ) -> dict[str, torch.Tensor]:
        scale = float(router_scale)
        action_inputs = torch.cat([a_action, pair_action, action_set_logits, shared_context], dim=-1)
        action_gate = self.action_router(action_inputs)
        action_residual = torch.tanh(pair_action + action_set_logits) * self.action_residual_cap
        final_action = a_action + scale * action_gate * action_residual
        calibrated_reason = r_reason + 0.1 * torch.tanh(self.reason_calibration(r_reason))
        reason_inputs = torch.cat([r_reason, pair_reason, calibrated_reason, reliability, shared_context], dim=-1)
        reason_gate = self.reason_router(reason_inputs)
        final_reason = r_reason + scale * reason_gate * reliability * torch.tanh(pair_reason + calibrated_reason - r_reason)
        return {
            "final_action_logits": final_action,
            "final_reason_logits": final_reason,
            "calibrated_reason_logits": calibrated_reason,
            "action_router_gate": action_gate,
            "reason_router_gate": reason_gate,
            "router_scale": final_action.new_tensor(scale),
            "action_residual": action_residual,
        }
