from __future__ import annotations

import torch
from torch import nn

from .types import InteractionDecisionLedger


class DecisionLedgerHead(nn.Module):
    def __init__(self, dim: int = 384, num_actions: int = 3) -> None:
        super().__init__()
        if num_actions != 3:
            raise ValueError("PSI CALI-Flow++ formal action_dim must be 3")
        self.num_actions = num_actions
        self.visual_head = nn.Linear(dim, num_actions)
        self.motion_head = nn.Linear(dim, num_actions)
        self.predicate_head = nn.Linear(dim, num_actions)
        self.flow_gate = nn.Linear(dim, num_actions)
        self.benefit_gate = nn.Linear(dim, num_actions)
        self.calibration_bias = nn.Parameter(torch.zeros(num_actions))
        self.global_hidden = nn.Linear(dim * 3, dim)

    @staticmethod
    def _soft_kl(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)
        target = soft_targets.clamp_min(1e-9)
        return (target * (target.log() - log_probs)).sum(-1)

    def forward(
        self,
        visual_token: torch.Tensor,
        motion_token: torch.Tensor,
        predicate_token: torch.Tensor,
        factor_tokens: torch.Tensor,
        flow_edges: torch.Tensor,
        action_soft_target: torch.Tensor | None = None,
    ) -> InteractionDecisionLedger:
        visual_logits = self.visual_head(visual_token)
        motion_logits = self.motion_head(motion_token)
        predicate_logits = self.predicate_head(predicate_token)
        gate = torch.sigmoid(self.flow_gate(factor_tokens)).clamp(0.05, 0.95)
        benefit = torch.sigmoid(self.benefit_gate(factor_tokens)).clamp(0.0, 1.0)
        global_logits = visual_logits + 0.35 * motion_logits + 0.25 * predicate_logits
        raw_state_contributions = flow_edges.clamp(-0.35, 0.35)
        benefit_target = None
        if action_soft_target is not None:
            with torch.no_grad():
                base_kl = self._soft_kl(global_logits, action_soft_target).unsqueeze(1)
                candidate_kl = self._soft_kl(global_logits.unsqueeze(1) + raw_state_contributions, action_soft_target.unsqueeze(1))
                advantage = base_kl - candidate_kl
                benefit_target = torch.sigmoid(advantage / 0.05).detach().unsqueeze(-1)
        gated_state_contributions = benefit * gate * raw_state_contributions
        flow_delta_logits = gated_state_contributions.sum(1)
        calibration_delta = self.calibration_bias.view(1, -1).expand_as(global_logits)
        final = global_logits + flow_delta_logits + calibration_delta
        reconstruction = global_logits + gated_state_contributions.sum(1) + calibration_delta
        identity_error = (final - reconstruction).abs().max()
        contribution_attention = gated_state_contributions.abs().sum(-1)
        contribution_attention = contribution_attention / contribution_attention.sum(-1, keepdim=True).clamp_min(1e-8)
        global_hidden = torch.tanh(self.global_hidden(torch.cat([visual_token, motion_token, predicate_token], dim=-1)))
        return InteractionDecisionLedger(
            global_logits=global_logits,
            visual_logits=visual_logits,
            motion_logits=motion_logits,
            predicate_logits=predicate_logits,
            raw_state_contributions=raw_state_contributions,
            gated_state_contributions=gated_state_contributions,
            flow_delta_logits=flow_delta_logits,
            calibration_delta=calibration_delta,
            final_logits=final,
            gate=gate,
            benefit_gate=benefit,
            benefit_target=benefit_target,
            contribution_attention=contribution_attention,
            global_hidden=global_hidden,
            contribution_terms={
                "visual": visual_logits,
                "motion": 0.35 * motion_logits,
                "predicate": 0.25 * predicate_logits,
                "flow": flow_delta_logits,
                "calibration": calibration_delta,
            },
            identity_error=identity_error,
        )
