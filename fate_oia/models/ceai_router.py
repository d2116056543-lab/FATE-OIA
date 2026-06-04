from __future__ import annotations

import torch
from torch import nn


class ParetoSafeRouter(nn.Module):
    def __init__(
        self,
        action_dim: int = 4,
        reason_dim: int = 21,
        action_cap: float = 0.04,
        reason_cap: float = 0.12,
        tail_reason_cap: float = 0.18,
        tail_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.action_cap = min(float(action_cap), 0.08)
        self.reason_cap = float(reason_cap)
        self.tail_reason_cap = float(tail_reason_cap)
        self.tail_indices = list(tail_indices or [5, 6, 9, 10, 11, 12, 13, 14])
        self.action_gate = nn.Sequential(nn.Linear(action_dim + action_dim + 2, action_dim), nn.Sigmoid())
        self.reason_gate = nn.Sequential(nn.Linear(reason_dim + reason_dim + reason_dim, reason_dim), nn.Sigmoid())

    def forward(
        self,
        base_action_logits: torch.Tensor,
        base_reason_logits: torch.Tensor,
        action_specialist_logits: torch.Tensor,
        reason_specialist_logits: torch.Tensor,
        pair_support: torch.Tensor,
        pair_reliability: torch.Tensor,
        reason_reliability: torch.Tensor | None = None,
        action_set_logits: torch.Tensor | None = None,
        tail_delta: torch.Tensor | None = None,
        readiness: dict | None = None,
    ) -> dict[str, torch.Tensor | dict[str, float] | str]:
        readiness = readiness or {}
        rel_a = pair_reliability.mean(dim=2)
        pair_action = (pair_support * pair_reliability).mean(dim=2)
        action_aux = action_set_logits if action_set_logits is not None else action_specialist_logits
        action_gate = self.action_gate(torch.cat([base_action_logits, action_aux, rel_a.mean(dim=1, keepdim=True), rel_a.std(dim=1, keepdim=True)], dim=-1))
        if bool(readiness.get("r2a_active", False)):
            action_delta = self.action_cap * torch.tanh(action_gate * (0.5 * (action_aux - base_action_logits) + 0.5 * pair_action))
        else:
            action_delta = base_action_logits.new_zeros(base_action_logits.shape)
        # Required anchor: final action is always base_action_logits + action_delta.
        final_action_logits = base_action_logits + action_delta

        q_r = reason_reliability if reason_reliability is not None else pair_reliability.max(dim=1).values
        reason_pair = (pair_support * pair_reliability).mean(dim=1)
        reason_gate = self.reason_gate(torch.cat([base_reason_logits, reason_specialist_logits, q_r], dim=-1))
        raw_reason_delta = reason_gate * (reason_specialist_logits - base_reason_logits + reason_pair)
        if tail_delta is not None:
            raw_reason_delta = raw_reason_delta + q_r * tail_delta
        cap = torch.full_like(raw_reason_delta, self.reason_cap)
        for idx in self.tail_indices:
            if 0 <= idx < cap.shape[1]:
                cap[:, idx] = self.tail_reason_cap
        reason_delta = cap * torch.tanh(raw_reason_delta)
        final_reason_logits = base_reason_logits + reason_delta
        stats = {
            "router_action_gate": float(action_gate.detach().mean().cpu()),
            "router_reason_gate": float(reason_gate.detach().mean().cpu()),
            "action_delta_abs_max": float(action_delta.detach().abs().max().cpu()),
            "reason_delta_abs_max": float(reason_delta.detach().abs().max().cpu()),
            "readiness_r2a": float(1.0 if bool(readiness.get("r2a_active", False)) else 0.0),
            "guarded_action_branch": "final",
        }
        return {
            "final_action_logits": final_action_logits,
            "final_reason_logits": final_reason_logits,
            "router_action_gate": action_gate,
            "router_reason_gate": reason_gate,
            "action_correction": action_delta,
            "reason_correction": reason_delta,
            "stats": stats,
            "guarded_action_branch": "final",
        }


def guarded_action_metrics(base_score: float, final_score: float, tolerance: float = 0.006) -> dict[str, str | float]:
    branch = "base" if final_score < base_score - tolerance else "final"
    return {"guarded_action_branch": branch, "base_score": float(base_score), "final_score": float(final_score), "tolerance": float(tolerance)}
