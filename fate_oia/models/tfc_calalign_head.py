from __future__ import annotations

import torch
from torch import nn


class TFCCalAlignHead(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        reason_dim: int = 21,
        action_threshold_min: float = 0.10,
        action_threshold_max: float = 0.90,
        reason_threshold_min: float = 0.02,
        reason_threshold_max: float = 0.85,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.action_base = nn.Parameter(torch.zeros(action_dim))
        self.reason_base = nn.Parameter(torch.zeros(reason_dim))
        self.action_delta = nn.Linear(action_dim * 3, action_dim)
        self.reason_delta = nn.Linear(reason_dim * 4, reason_dim)
        nn.init.zeros_(self.action_delta.weight); nn.init.zeros_(self.action_delta.bias)
        nn.init.zeros_(self.reason_delta.weight); nn.init.zeros_(self.reason_delta.bias)
        self.register_buffer("action_min", torch.logit(torch.full((action_dim,), action_threshold_min)))
        self.register_buffer("action_max", torch.logit(torch.full((action_dim,), action_threshold_max)))
        self.register_buffer("reason_min", torch.logit(torch.full((reason_dim,), reason_threshold_min)))
        self.register_buffer("reason_max", torch.logit(torch.full((reason_dim,), reason_threshold_max)))

    def forward(
        self,
        action_logits: torch.Tensor,
        reason_logits: torch.Tensor,
        credit_confidence_action: torch.Tensor,
        credit_confidence_reason: torch.Tensor,
        action_margins: torch.Tensor | None = None,
        reason_support: torch.Tensor | None = None,
        reason_contra: torch.Tensor | None = None,
        reason_rho: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b = action_logits.shape[0]
        action_margins = torch.zeros_like(action_logits) if action_margins is None else action_margins.detach()
        reason_support = torch.zeros_like(reason_logits) if reason_support is None else reason_support.detach()
        reason_contra = torch.zeros_like(reason_logits) if reason_contra is None else reason_contra.detach()
        reason_rho = torch.zeros_like(reason_logits) if reason_rho is None else reason_rho.detach()
        action_inputs = torch.cat([credit_confidence_action.detach(), action_margins.detach(), action_logits.detach().abs()], dim=-1)
        reason_inputs = torch.cat([credit_confidence_reason.detach(), reason_support, reason_contra, reason_rho], dim=-1)
        theta_delta_action = 0.25 * torch.tanh(self.action_delta(action_inputs))
        theta_delta_reason = 0.25 * torch.tanh(self.reason_delta(reason_inputs))
        action_theta = (self.action_base.view(1, -1) + theta_delta_action).clamp(self.action_min.view(1, -1), self.action_max.view(1, -1))
        reason_theta = (self.reason_base.view(1, -1) + theta_delta_reason).clamp(self.reason_min.view(1, -1), self.reason_max.view(1, -1))
        action_logits_deploy = action_logits - action_theta
        reason_logits_deploy = reason_logits - reason_theta
        return {
            "action_theta": action_theta,
            "reason_theta": reason_theta,
            "action_logits_deploy": action_logits_deploy,
            "reason_logits_deploy": reason_logits_deploy,
            "logits_deploy": torch.cat([action_logits_deploy, reason_logits_deploy], dim=-1),
            "theta_delta_action": theta_delta_action,
            "theta_delta_reason": theta_delta_reason,
            "threshold_input_stopgrad_check": torch.tensor(True, device=action_logits.device),
        }
