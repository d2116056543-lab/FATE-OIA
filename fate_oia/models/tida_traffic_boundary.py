from __future__ import annotations

import torch
from torch import nn


class TIDATrafficAdaptiveBoundary(nn.Module):
    """Bounded per-action deployment boundary from detached traffic evidence."""

    def __init__(self, num_actions: int = 4, state_dim: int = 8, hidden_dim: int = 32, cap: float = 0.25) -> None:
        super().__init__()
        if num_actions <= 0 or state_dim <= 0 or hidden_dim <= 0 or cap <= 0:
            raise ValueError("invalid adaptive boundary dimensions")
        self.num_actions = int(num_actions)
        self.state_dim = int(state_dim)
        self.cap = float(cap)
        self.action_embedding = nn.Parameter(torch.randn(num_actions, 8) * 0.02)
        self.network = nn.Sequential(
            nn.LayerNorm(state_dim + 6 + 8),
            nn.Linear(state_dim + 6 + 8, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(
        self,
        base_logits: torch.Tensor,
        order_delta: torch.Tensor,
        state_features: torch.Tensor,
        support: torch.Tensor,
        state_strength: torch.Tensor,
        interaction_risk: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, actions = base_logits.shape
        if actions != self.num_actions or order_delta.shape != base_logits.shape:
            raise ValueError("base/order tensors must be [B,A]")
        if state_features.shape != (batch, actions, self.state_dim):
            raise ValueError("state_features must be [B,A,S]")
        for name, value in (("support", support), ("state_strength", state_strength)):
            if value.shape != base_logits.shape:
                raise ValueError(f"{name} must be [B,A]")
        if interaction_risk.ndim != 3 or interaction_risk.shape[:2] != base_logits.shape:
            raise ValueError("interaction_risk must be [B,A,K]")
        detached_base = base_logits.detach()
        scalar = torch.stack(
            (
                detached_base,
                detached_base.abs(),
                order_delta.detach(),
                support.detach(),
                state_strength.detach(),
                interaction_risk.detach().mean(-1),
            ),
            dim=-1,
        )
        identity = self.action_embedding[None].expand(batch, -1, -1)
        features = torch.cat((state_features.detach(), scalar, identity), dim=-1)
        boundary_delta = self.cap * torch.tanh(self.network(features).squeeze(-1))
        return {
            "traffic_adaptive_boundary_delta": boundary_delta,
            "traffic_adaptive_deploy_action_logits": base_logits - boundary_delta,
            "traffic_adaptive_boundary_features": features,
        }
