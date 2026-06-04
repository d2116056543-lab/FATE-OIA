from __future__ import annotations

import torch
from torch import nn


class ParetoSafeRouter(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        action_residual_cap: float = 0.04,
        temperature_action: float = 1.5,
        temperature_reason: float = 1.5,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.action_residual_cap = float(action_residual_cap)
        self.temperature_action = float(temperature_action)
        self.temperature_reason = float(temperature_reason)
        self.action_router = nn.Sequential(nn.Linear(action_dim * 3 + dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, action_dim), nn.Sigmoid())
        self.reason_router_body = nn.Sequential(nn.Linear(reason_dim * 4 + dim, dim), nn.GELU(), nn.Linear(dim, reason_dim))
        self.reason_calibration = nn.Linear(reason_dim, reason_dim)

    def forward(
        self,
        base_action: torch.Tensor,
        base_reason: torch.Tensor,
        action_specialist: torch.Tensor,
        reason_specialist: torch.Tensor,
        pair_action: torch.Tensor,
        pair_reason: torch.Tensor,
        action_set_logits: torch.Tensor,
        reliability: torch.Tensor,
        shared_context: torch.Tensor,
        action_router_scale: float,
        reason_router_scale: float,
    ) -> dict[str, torch.Tensor]:
        action_scale = float(action_router_scale)
        reason_scale = float(reason_router_scale)
        action_inputs = torch.cat([action_specialist, pair_action, action_set_logits, shared_context], dim=-1)
        action_gate = self.action_router(action_inputs / max(self.temperature_action, 1e-6))
        action_residual = torch.tanh(pair_action + action_set_logits) * self.action_residual_cap
        final_action = base_action + action_scale * action_gate * action_residual
        calibrated_reason = base_reason + 0.1 * torch.tanh(self.reason_calibration(reason_specialist))
        reason_inputs = torch.cat([reason_specialist, pair_reason, calibrated_reason, reliability, shared_context], dim=-1)
        reason_gate = torch.sigmoid(self.reason_router_body(reason_inputs) / max(self.temperature_reason, 1e-6))
        final_reason = base_reason + reason_scale * reason_gate * reliability * torch.tanh(pair_reason + calibrated_reason - base_reason)
        action_entropy = -(action_gate.clamp(1e-6, 1 - 1e-6) * action_gate.clamp(1e-6, 1 - 1e-6).log() + (1 - action_gate).clamp(1e-6, 1 - 1e-6) * (1 - action_gate).clamp(1e-6, 1 - 1e-6).log()).mean()
        reason_entropy = -(reason_gate.clamp(1e-6, 1 - 1e-6) * reason_gate.clamp(1e-6, 1 - 1e-6).log() + (1 - reason_gate).clamp(1e-6, 1 - 1e-6) * (1 - reason_gate).clamp(1e-6, 1 - 1e-6).log()).mean()
        return {
            "final_action_logits": final_action,
            "final_reason_logits": final_reason,
            "calibrated_reason_logits": calibrated_reason,
            "action_router_gate": action_gate,
            "reason_router_gate": reason_gate,
            "action_router_scale": final_action.new_tensor(action_scale),
            "reason_router_scale": final_action.new_tensor(reason_scale),
            "router_scale": final_action.new_tensor(max(action_scale, reason_scale)),
            "action_residual": action_residual,
            "action_gate_entropy": action_entropy,
            "reason_gate_entropy": reason_entropy,
            "gate_entropy": 0.5 * (action_entropy + reason_entropy),
        }
