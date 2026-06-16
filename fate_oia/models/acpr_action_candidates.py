from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ACPRActionCandidates(nn.Module):
    """Candidate-probe action utility layer.

    The current ACPR-CalAlign deploy logits remain the fallback. Candidate
    logits are trained/evaluated separately; final utility only changes an
    action dimension when a train-calib gate selects a candidate for it.
    """

    def __init__(
        self,
        action_dim: int = 4,
        max_pred_delta: float = 0.05,
        initial_blend_gamma: float = 0.5,
        candidate_names: tuple[str, ...] = ("visual", "reason", "blend", "predicate", "blend_predicate"),
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.max_pred_delta = float(max_pred_delta)
        self.candidate_names = tuple(candidate_names)
        gamma = torch.full((self.action_dim,), float(initial_blend_gamma)).clamp(1e-4, 1.0 - 1e-4)
        self.blend_gamma_raw = nn.Parameter(torch.logit(gamma))
        self.register_buffer("selected_candidate_id", torch.full((self.action_dim,), -1, dtype=torch.long))
        self.register_buffer("selected_gate", torch.zeros(self.action_dim, dtype=torch.float32))

    def clear_selected_candidates(self) -> None:
        self.selected_candidate_id.fill_(-1)
        self.selected_gate.zero_()

    def set_selected_candidates(self, candidate_ids: torch.Tensor, gates: torch.Tensor) -> None:
        ids = candidate_ids.detach().to(self.selected_candidate_id.device).long().view(self.action_dim)
        gate = gates.detach().to(self.selected_gate.device, self.selected_gate.dtype).view(self.action_dim)
        valid = (ids >= 0) & (ids < len(self.candidate_names)) & (gate > 0)
        self.selected_candidate_id.copy_(torch.where(valid, ids, torch.full_like(ids, -1)))
        self.selected_gate.copy_(torch.where(valid, gate.clamp(0.0, 1.0), torch.zeros_like(gate)))

    def selected_summary(self) -> dict[str, Any]:
        ids = self.selected_candidate_id.detach().cpu().tolist()
        gates = self.selected_gate.detach().cpu().tolist()
        return {
            "candidate_names": list(self.candidate_names),
            "selected_candidate_id": ids,
            "selected_candidate_name": [
                self.candidate_names[int(i)] if 0 <= int(i) < len(self.candidate_names) else "fallback"
                for i in ids
            ],
            "selected_gate": gates,
        }

    def forward(
        self,
        action_logits_fallback: torch.Tensor,
        action_visual_logits: torch.Tensor,
        action_reason_logits: torch.Tensor,
        theta_action: torch.Tensor,
        predicate_action_delta: torch.Tensor | None = None,
        probe_mode: bool = False,
    ) -> dict[str, Any]:
        del probe_mode  # candidates are always returned; final still obeys selected gates.
        theta = theta_action.to(action_logits_fallback.device, action_logits_fallback.dtype).view(1, self.action_dim)
        visual = action_visual_logits - theta
        reason = action_reason_logits - theta
        gamma = torch.sigmoid(self.blend_gamma_raw).to(action_logits_fallback.device, action_logits_fallback.dtype)
        blend_base = (1.0 - gamma.view(1, -1)) * action_visual_logits + gamma.view(1, -1) * action_reason_logits
        blend = blend_base - theta
        if predicate_action_delta is None:
            predicate_delta = torch.zeros_like(action_logits_fallback)
        else:
            predicate_delta = predicate_action_delta.to(action_logits_fallback.device, action_logits_fallback.dtype).clamp(
                -self.max_pred_delta, self.max_pred_delta
            )
        candidates = {
            "fallback": action_logits_fallback,
            "visual": visual,
            "reason": reason,
            "blend": blend,
            "predicate": action_logits_fallback + predicate_delta,
            "blend_predicate": blend + predicate_delta,
        }
        utility = action_logits_fallback.clone()
        ids = self.selected_candidate_id.to(action_logits_fallback.device)
        gates = self.selected_gate.to(action_logits_fallback.device, action_logits_fallback.dtype).clamp(0.0, 1.0)
        for action_idx in range(self.action_dim):
            cid = int(ids[action_idx].item())
            gate = gates[action_idx]
            if cid >= 0 and cid < len(self.candidate_names) and float(gate.item()) > 0.0:
                name = self.candidate_names[cid]
                utility[:, action_idx] = (1.0 - gate) * action_logits_fallback[:, action_idx] + gate * candidates[name][:, action_idx]
        return {
            **candidates,
            "utility_final": utility,
            "predicate_delta_clipped": predicate_delta,
            "blend_gamma": gamma,
            "selected_candidate_id": self.selected_candidate_id,
            "selected_gate": self.selected_gate,
            "candidate_names": list(self.candidate_names),
        }
