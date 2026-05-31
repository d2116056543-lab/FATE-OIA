from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class ActionPrimaryConflictGate:
    conflict_threshold: float = -0.02
    downscale_reason_min: float = 0.35
    downscale_evidence_min: float = 0.20
    action_floor_epoch: int = 6
    action_floor_mF1: float = 0.700
    if_below_floor_evidence_scale: float = 0.50
    if_below_floor_counterfactual_scale: float = 0.0

    @classmethod
    def from_args(cls, args) -> "ActionPrimaryConflictGate":
        return cls(
            conflict_threshold=float(getattr(args, "conflict_threshold", -0.02)),
            downscale_reason_min=float(getattr(args, "downscale_reason_min", 0.35)),
            downscale_evidence_min=float(getattr(args, "downscale_evidence_min", 0.20)),
            action_floor_epoch=int(getattr(args, "action_floor_epoch", 6)),
            action_floor_mF1=float(getattr(args, "action_floor_mF1", 0.700)),
            if_below_floor_evidence_scale=float(getattr(args, "if_below_floor_evidence_scale", 0.50)),
            if_below_floor_counterfactual_scale=float(getattr(args, "if_below_floor_counterfactual_scale", 0.0)),
        )

    def _flat_grad(self, loss: torch.Tensor, params: list[torch.nn.Parameter]) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        chunks = []
        for param, grad in zip(params, grads):
            chunks.append((torch.zeros_like(param) if grad is None else grad.detach()).reshape(-1))
        return torch.cat(chunks) if chunks else loss.detach().new_zeros((1,))

    @staticmethod
    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        denom = float(a.norm().item() * b.norm().item())
        if denom <= 1e-12:
            return 0.0
        return float(torch.dot(a, b).item() / denom)

    def _scale(self, cosine: float, min_scale: float) -> float:
        if cosine < self.conflict_threshold:
            return max(float(min_scale), 1.0 + float(cosine))
        return 1.0

    def compute(
        self,
        action_loss: torch.Tensor,
        reason_loss: torch.Tensor,
        evidence_loss: torch.Tensor,
        sentinel_params: Iterable[torch.nn.Parameter],
        epoch: int,
        latest_act_mf1: float | None,
    ) -> dict[str, float | bool]:
        params = [p for p in sentinel_params if p.requires_grad]
        if not params:
            return {
                "grad_cos_action_reason": 0.0,
                "grad_cos_action_evidence": 0.0,
                "applied_reason_scale": 1.0,
                "applied_evidence_scale": 1.0,
                "applied_counterfactual_scale": 1.0,
                "action_floor_active": False,
            }
        ga = self._flat_grad(action_loss, params)
        gr = self._flat_grad(reason_loss, params)
        ge = self._flat_grad(evidence_loss, params)
        cr = self._cos(ga, gr)
        ce = self._cos(ga, ge)
        reason_scale = self._scale(cr, self.downscale_reason_min)
        evidence_scale = self._scale(ce, self.downscale_evidence_min)
        cf_scale = 1.0
        floor_active = False
        if latest_act_mf1 is not None and int(epoch) >= self.action_floor_epoch and float(latest_act_mf1) < self.action_floor_mF1:
            floor_active = True
            evidence_scale *= self.if_below_floor_evidence_scale
            cf_scale = self.if_below_floor_counterfactual_scale
        return {
            "grad_cos_action_reason": cr,
            "grad_cos_action_evidence": ce,
            "applied_reason_scale": float(reason_scale),
            "applied_evidence_scale": float(evidence_scale),
            "applied_counterfactual_scale": float(cf_scale),
            "action_floor_active": bool(floor_active),
        }
