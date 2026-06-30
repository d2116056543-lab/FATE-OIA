from __future__ import annotations

import torch
from torch import nn


class NativeTextMetricCalibrator(nn.Module):
    def __init__(self, action_dim: int = 4, reason_dim: int = 21, delta_cap_reason: float = 0.15, delta_cap_action: float = 0.05) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.theta_action_global = nn.Parameter(torch.zeros(action_dim))
        self.theta_reason_global = nn.Parameter(torch.zeros(reason_dim))
        self.register_buffer("teacher_theta_action", torch.zeros(action_dim))
        self.register_buffer("teacher_theta_reason", torch.zeros(reason_dim))
        self.register_buffer("teacher_epoch", torch.tensor(-1, dtype=torch.long))
        self.reason_delta = nn.Sequential(nn.Linear(5, 32), nn.GELU(), nn.Linear(32, 1))
        self.action_delta = nn.Sequential(nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1))
        self.delta_cap_reason = float(delta_cap_reason)
        self.delta_cap_action = float(delta_cap_action)

    @torch.no_grad()
    def update_teacher(self, action_threshold_logit: torch.Tensor, reason_threshold_logit: torch.Tensor, epoch: int) -> None:
        self.teacher_theta_action.copy_(action_threshold_logit.detach().to(self.teacher_theta_action.device, self.teacher_theta_action.dtype))
        self.teacher_theta_reason.copy_(reason_threshold_logit.detach().to(self.teacher_theta_reason.device, self.teacher_theta_reason.dtype))
        self.teacher_epoch.fill_(int(epoch))

    def forward(
        self,
        action_logits: torch.Tensor,
        reason_logits: torch.Tensor,
        support_score: torch.Tensor,
        contra_score: torch.Tensor,
        reason_rho: torch.Tensor,
        base_reason_logits: torch.Tensor,
        q_pred: torch.Tensor,
        rho_pred: torch.Tensor,
        epoch: int = 0,
    ) -> dict[str, torch.Tensor | dict]:
        margin = reason_logits.detach().abs()
        card = torch.sigmoid(reason_logits.detach()).sum(-1, keepdim=True).expand_as(reason_logits) / max(self.reason_dim, 1)
        rfeat = torch.stack([support_score.detach(), contra_score.detach(), reason_rho.detach(), margin, card], dim=-1)
        cap_r = 0.02 if epoch < 3 else min(self.delta_cap_reason, 0.03 + 0.03 * epoch)
        threshold_delta_reason = torch.tanh(self.reason_delta(rfeat).squeeze(-1)) * cap_r
        pmean = (q_pred * rho_pred).mean(-1, keepdim=True).expand(-1, self.action_dim).detach()
        amargin = action_logits.detach().abs()
        acard = torch.sigmoid(action_logits.detach()).sum(-1, keepdim=True).expand_as(action_logits) / max(self.action_dim, 1)
        afeat = torch.stack([pmean, amargin, acard], dim=-1)
        cap_a = min(self.delta_cap_action, 0.01 + 0.01 * epoch)
        threshold_delta_action = torch.tanh(self.action_delta(afeat).squeeze(-1)) * cap_a
        theta_reason = self.theta_reason_global.view(1, -1) + threshold_delta_reason
        theta_action = self.theta_action_global.view(1, -1) + threshold_delta_action
        return {
            "theta_action": theta_action,
            "theta_reason": theta_reason,
            "theta_action_global": self.theta_action_global,
            "theta_reason_global": self.theta_reason_global,
            "theta_action_teacher": self.teacher_theta_action,
            "theta_reason_teacher": self.teacher_theta_reason,
            "threshold_delta_reason": threshold_delta_reason,
            "threshold_delta_action": threshold_delta_action,
            "action_logits_deploy": action_logits - theta_action,
            "reason_logits_deploy": reason_logits - theta_reason,
            "logits_deploy": torch.cat([action_logits - theta_action, reason_logits - theta_reason], dim=-1),
            "threshold_stats": {
                "threshold_delta_reason_abs_mean": float(threshold_delta_reason.abs().mean().detach().cpu()),
                "threshold_delta_action_abs_mean": float(threshold_delta_action.abs().mean().detach().cpu()),
                "teacher_epoch": int(self.teacher_epoch.item()),
            },
        }
