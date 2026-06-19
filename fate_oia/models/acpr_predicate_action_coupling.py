from __future__ import annotations

import torch
from torch import nn


class ACPRPredicateActionCoupling(nn.Module):
    """Route predicate-conditioned reason evidence into the action reason branch.

    This module is deliberately bounded and transparent. It does not create an
    mixture/router path and it never uses action-set marginals. The only action
    correction is the existing reason-to-action projection applied to the same
    predicate-conditioned reason delta used by the explanation logits.
    """

    def __init__(
        self,
        action_dim: int = 4,
        reason_dim: int = 21,
        coupling_strength: float = 1.0,
        max_action_delta: float = 0.20,
        epsilon: float = 1.0e-8,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.coupling_strength = float(coupling_strength)
        self.max_action_delta = float(max_action_delta)
        self.epsilon = float(epsilon)

    def forward(
        self,
        action_visual_logits: torch.Tensor,
        action_reason_logits_visual: torch.Tensor,
        action_fusion_gate: torch.Tensor,
        predicate_reason_delta: torch.Tensor,
        reason_to_action_weight: torch.Tensor,
        reason_to_action_bias: torch.Tensor | None = None,
        coupling_strength: float | None = None,
    ) -> dict[str, torch.Tensor]:
        del reason_to_action_bias
        kappa = self.coupling_strength if coupling_strength is None else float(coupling_strength)
        legacy = action_fusion_gate * action_visual_logits + (1.0 - action_fusion_gate) * action_reason_logits_visual
        raw_contrib = reason_to_action_weight.to(predicate_reason_delta.device, predicate_reason_delta.dtype).unsqueeze(0) * predicate_reason_delta.unsqueeze(1)
        raw_delta = raw_contrib.sum(-1)
        if kappa == 0.0:
            bounded = torch.zeros_like(raw_delta)
        else:
            bounded = self.max_action_delta * torch.tanh(kappa * raw_delta / max(self.max_action_delta, self.epsilon))
        abs_raw = raw_delta.abs()
        safe_scale = torch.where(abs_raw > self.epsilon, bounded / raw_delta.clamp(min=-1.0e12, max=1.0e12), torch.full_like(raw_delta, float(kappa)))
        bounded_reason_contrib = raw_contrib * safe_scale.unsqueeze(-1)
        action_reason_pace = action_reason_logits_visual + bounded
        action_pace = action_fusion_gate * action_visual_logits + (1.0 - action_fusion_gate) * action_reason_pace
        final_reason_contrib = (1.0 - action_fusion_gate).unsqueeze(-1) * bounded_reason_contrib
        return {
            "action_logits_legacy": legacy,
            "action_reason_logits_visual": action_reason_logits_visual,
            "action_reason_logits_pace": action_reason_pace,
            "action_logits_pace": action_pace,
            "predicate_action_delta_raw": raw_delta,
            "predicate_action_delta_bounded": bounded,
            "predicate_reason_action_contrib_raw": raw_contrib,
            "predicate_reason_action_contrib_final": final_reason_contrib,
            "pace_coupling_strength": torch.tensor(float(kappa), device=raw_delta.device, dtype=raw_delta.dtype),
            "pace_max_action_delta": torch.tensor(float(self.max_action_delta), device=raw_delta.device, dtype=raw_delta.dtype),
            "pace_action_delta_abs_mean": bounded.abs().mean(),
            "pace_action_delta_per_action_mean": bounded.mean(0),
            "pace_action_delta_saturation_rate": (abs_raw > self.max_action_delta).float().mean(),
        }
