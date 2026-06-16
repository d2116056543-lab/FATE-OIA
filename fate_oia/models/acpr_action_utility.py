from __future__ import annotations

import torch
from torch import nn


class ACPRActionUtility(nn.Module):
    """Guarded action utility head with zero-gate fallback equality.

    utility = fallback + r2a_gate * clamp(action_reason - action_visual)
                       + pred_gate * clamp(predicate_delta)
    Gates are buffers updated from train_calib only. With both gates equal to
    zero, action_logits_utility is exactly action_logits_fallback.
    """

    def __init__(
        self,
        action_dim: int = 4,
        max_r2a_delta: float = 0.20,
        max_pred_delta: float = 0.05,
        initial_r2a_gate: float = 0.0,
        initial_pred_gate: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.max_r2a_delta = float(max_r2a_delta)
        self.max_pred_delta = float(max_pred_delta)
        self.register_buffer("r2a_gate", torch.full((self.action_dim,), float(initial_r2a_gate)))
        self.register_buffer("pred_gate", torch.full((self.action_dim,), float(initial_pred_gate)))

    def set_gates(self, r2a_gate: torch.Tensor | None = None, pred_gate: torch.Tensor | None = None) -> None:
        if r2a_gate is not None:
            self.r2a_gate.copy_(r2a_gate.detach().to(self.r2a_gate.device, self.r2a_gate.dtype).clamp(0.0, 1.0))
        if pred_gate is not None:
            self.pred_gate.copy_(pred_gate.detach().to(self.pred_gate.device, self.pred_gate.dtype).clamp(0.0, 1.0))

    def forward(
        self,
        action_logits_fallback: torch.Tensor,
        action_visual_logits: torch.Tensor,
        action_reason_logits: torch.Tensor,
        predicate_action_delta: torch.Tensor | None = None,
        override_r2a_gate: torch.Tensor | None = None,
        override_pred_gate: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        r2a_delta = (action_reason_logits - action_visual_logits).clamp(-self.max_r2a_delta, self.max_r2a_delta)
        pred_delta = torch.zeros_like(action_logits_fallback) if predicate_action_delta is None else predicate_action_delta.clamp(-self.max_pred_delta, self.max_pred_delta)
        r2a_gate = self.r2a_gate if override_r2a_gate is None else override_r2a_gate
        pred_gate = self.pred_gate if override_pred_gate is None else override_pred_gate
        r2a_gate = r2a_gate.to(action_logits_fallback.device, action_logits_fallback.dtype).clamp(0.0, 1.0).view(1, -1)
        pred_gate = pred_gate.to(action_logits_fallback.device, action_logits_fallback.dtype).clamp(0.0, 1.0).view(1, -1)
        utility = action_logits_fallback + r2a_gate * r2a_delta + pred_gate * pred_delta
        return {
            "action_logits_fallback": action_logits_fallback,
            "action_logits_utility": utility,
            "action_r2a_delta": r2a_delta,
            "action_predicate_delta": pred_delta,
            "r2a_gate": r2a_gate.squeeze(0),
            "pred_gate": pred_gate.squeeze(0),
            "r2a_delta_abs_mean": r2a_delta.abs().mean(),
            "pred_delta_abs_mean": pred_delta.abs().mean(),
        }
