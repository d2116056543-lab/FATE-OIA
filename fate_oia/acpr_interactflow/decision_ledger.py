from __future__ import annotations

import torch
from torch import nn

from .types import InteractionDecisionLedger


class DecisionLedgerHead(nn.Module):
    def __init__(self, dim: int = 384, num_actions: int = 4) -> None:
        super().__init__()
        self.visual_head = nn.Linear(dim, num_actions)
        self.motion_head = nn.Linear(dim, num_actions)
        self.predicate_head = nn.Linear(dim, num_actions)
        self.flow_gate = nn.Linear(dim, num_actions)
        self.benefit_gate = nn.Linear(dim, num_actions)
        self.calibration_bias = nn.Parameter(torch.zeros(num_actions))

    def forward(self, visual_token: torch.Tensor, motion_token: torch.Tensor, predicate_token: torch.Tensor, factor_tokens: torch.Tensor, flow_edges: torch.Tensor) -> InteractionDecisionLedger:
        visual_logits = self.visual_head(visual_token)
        motion_logits = self.motion_head(motion_token)
        predicate_logits = self.predicate_head(predicate_token)
        gate = torch.sigmoid(self.flow_gate(factor_tokens)).clamp(0.05, 0.95)
        benefit = torch.sigmoid(self.benefit_gate(factor_tokens)).clamp(0.0, 1.0)
        global_logits = visual_logits + 0.35 * motion_logits + 0.25 * predicate_logits
        gated_state_contributions = benefit * gate * flow_edges.clamp(-0.35, 0.35)
        flow_delta_logits = gated_state_contributions.sum(1)
        calibration_delta = self.calibration_bias.view(1, -1).expand_as(global_logits)
        final = global_logits + flow_delta_logits + calibration_delta
        reconstruction = global_logits + gated_state_contributions.sum(1) + calibration_delta
        identity_error = (final - reconstruction).abs().max()
        return InteractionDecisionLedger(
            global_logits=global_logits,
            visual_logits=visual_logits,
            motion_logits=motion_logits,
            predicate_logits=predicate_logits,
            gated_state_contributions=gated_state_contributions,
            flow_delta_logits=flow_delta_logits,
            calibration_delta=calibration_delta,
            final_logits=final,
            gate=gate,
            benefit_gate=benefit,
            contribution_terms={
                "visual": visual_logits,
                "motion": 0.35 * motion_logits,
                "predicate": 0.25 * predicate_logits,
                "flow": flow_delta_logits,
                "calibration": calibration_delta,
            },
            identity_error=identity_error,
        )
