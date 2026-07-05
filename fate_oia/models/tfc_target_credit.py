from __future__ import annotations

import torch
from torch import nn


class TFCTargetCredit(nn.Module):
    def __init__(self, num_factors: int, action_dim: int = 4, reason_dim: int = 21, dim: int = 384) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.feature_proj = nn.Linear(dim, dim)
        self.action_target_embeddings = nn.Parameter(torch.randn(action_dim, dim) * 0.02)
        self.reason_target_embeddings = nn.Parameter(torch.randn(reason_dim, dim) * 0.02)
        self.learned_action = nn.Parameter(torch.zeros(num_factors, action_dim))
        self.learned_reason = nn.Parameter(torch.zeros(num_factors, reason_dim))

    def forward(
        self,
        factor_probs: torch.Tensor,
        factor_rho: torch.Tensor,
        factor_features: torch.Tensor,
        compatibility: dict[str, torch.Tensor],
        action_margins: torch.Tensor | None = None,
        reason_margins: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        qrho = factor_probs * factor_rho
        native_action = (compatibility["factor_to_action_support"] - compatibility["factor_to_action_inhibit"]).to(qrho.device, qrho.dtype)
        native_reason = (compatibility["factor_to_reason_support"] - compatibility["factor_to_reason_inhibit"]).to(qrho.device, qrho.dtype)
        feat = torch.nn.functional.normalize(self.feature_proj(factor_features), dim=-1)
        action_embed = torch.nn.functional.normalize(self.action_target_embeddings.to(qrho.device, qrho.dtype), dim=-1)
        reason_embed = torch.nn.functional.normalize(self.reason_target_embeddings.to(qrho.device, qrho.dtype), dim=-1)
        instance_action_compat = torch.einsum("bfd,ad->bfa", feat, action_embed).tanh()
        instance_reason_compat = torch.einsum("bfd,rd->bfr", feat, reason_embed).tanh()
        learned_action = torch.tanh(self.learned_action).to(qrho.device, qrho.dtype).unsqueeze(0) + 0.25 * instance_action_compat
        learned_reason = torch.tanh(self.learned_reason).to(qrho.device, qrho.dtype).unsqueeze(0) + 0.25 * instance_reason_compat
        action_scale = (1.0 + 0.25 * learned_action).clamp(0.5, 1.5)
        reason_scale = (1.0 + 0.25 * learned_reason).clamp(0.5, 1.5)
        if action_margins is None:
            action_gate = torch.ones(qrho.shape[0], self.action_dim, device=qrho.device, dtype=qrho.dtype)
        else:
            action_gate = torch.sigmoid(1.0 - action_margins.detach().abs())
        if reason_margins is None:
            reason_gate = torch.ones(qrho.shape[0], self.reason_dim, device=qrho.device, dtype=qrho.dtype)
        else:
            reason_gate = torch.sigmoid(1.0 - reason_margins.detach().abs())
        # Native compatibility is the causal sign/mask. Learned terms may only
        # modulate known support/inhibit relations, never create residual credit
        # for unknown factor-target pairs.
        credit_action = qrho.unsqueeze(-1) * native_action.unsqueeze(0) * action_scale * action_gate.unsqueeze(1)
        credit_reason = qrho.unsqueeze(-1) * native_reason.unsqueeze(0) * reason_scale * reason_gate.unsqueeze(1)
        credit_action_norm = credit_action / (credit_action.abs().sum(dim=1, keepdim=True) + 1e-6)
        credit_reason_norm = credit_reason / (credit_reason.abs().sum(dim=1, keepdim=True) + 1e-6)
        return {
            "credit_action": credit_action,
            "credit_reason": credit_reason,
            "credit_action_norm": credit_action_norm,
            "credit_reason_norm": credit_reason_norm,
            "credit_confidence_action": credit_action.abs().sum(dim=1),
            "credit_confidence_reason": credit_reason.abs().sum(dim=1),
        }
