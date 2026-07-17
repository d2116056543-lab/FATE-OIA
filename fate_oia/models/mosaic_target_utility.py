"""Audit-derived target utility state for CREDO factor transport."""

from __future__ import annotations

import torch
from torch import nn


class MOSAICAuditTargetUtility(nn.Module):
    """Keep reason compatibility and action utility in disjoint audit state.

    The state is deliberately not predicted from a current training/test batch.
    It is updated only after an ``audit_target`` intervention run and consumed by
    the following epoch. The neutral pre-audit value is one, preserving learning
    access; deployment remains separately protected by edge admission.
    """

    def __init__(self, *, factor_count: int, reason_count: int = 21, action_count: int = 4, ema_decay: float = 0.80) -> None:
        super().__init__()
        if factor_count <= 0 or reason_count != 21 or action_count != 4:
            raise ValueError("CREDO target utility requires [21,F] reasons and [F,4] actions")
        if not 0.0 <= float(ema_decay) < 1.0:
            raise ValueError("target utility EMA decay must be in [0,1)")
        self.factor_count = int(factor_count)
        self.ema_decay = float(ema_decay)
        self.register_buffer("semantic_compatibility", torch.ones(reason_count, factor_count), persistent=True)
        self.register_buffer("action_target_utility", torch.ones(factor_count, action_count), persistent=True)
        self.register_buffer("target_utility_initialized", torch.tensor(False), persistent=True)

    @torch.no_grad()
    def update_from_audit(
        self,
        semantic_compatibility: torch.Tensor,
        action_target_utility: torch.Tensor,
        *,
        source_split: str,
    ) -> None:
        if source_split != "audit_target":
            raise ValueError("CREDO target utility only accepts audit_target evidence")
        if semantic_compatibility.shape != self.semantic_compatibility.shape:
            raise ValueError("semantic compatibility must be [21,F]")
        if action_target_utility.shape != self.action_target_utility.shape:
            raise ValueError("action target utility must be [F,4]")
        if not torch.isfinite(semantic_compatibility).all() or not torch.isfinite(action_target_utility).all():
            raise ValueError("target utility audit values must be finite")
        semantic = semantic_compatibility.detach().to(self.semantic_compatibility).clamp(0.0, 1.0)
        action = action_target_utility.detach().to(self.action_target_utility).clamp(0.0, 1.0)
        if bool(self.target_utility_initialized):
            semantic = self.ema_decay * self.semantic_compatibility + (1.0 - self.ema_decay) * semantic
            action = self.ema_decay * self.action_target_utility + (1.0 - self.ema_decay) * action
        self.semantic_compatibility.copy_(semantic)
        self.action_target_utility.copy_(action)
        self.target_utility_initialized.fill_(True)

    def forward(self) -> dict[str, torch.Tensor]:
        return {
            "semantic_compatibility": self.semantic_compatibility,
            "action_target_utility": self.action_target_utility,
            "target_utility_initialized": self.target_utility_initialized.to(dtype=torch.long),
        }
