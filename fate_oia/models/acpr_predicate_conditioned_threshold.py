from __future__ import annotations

import torch
from torch import nn

from .acpr_threshold_head import ACPRThresholdHead


class ACPRPredicateConditionedThreshold(ACPRThresholdHead):
    def __init__(self, action_dim=4, reason_dim=21, num_predicates=32, threshold_delta_max=0.10, **kwargs):
        super().__init__(action_dim=action_dim, reason_dim=reason_dim, **kwargs)
        self.num_predicates = int(num_predicates)
        self.threshold_delta_max = float(threshold_delta_max)
        self.predicate_to_delta = nn.Sequential(
            nn.Linear(self.num_predicates, 64),
            nn.GELU(),
            nn.Linear(64, self.num_labels),
        )
        nn.init.zeros_(self.predicate_to_delta[-1].weight)
        nn.init.zeros_(self.predicate_to_delta[-1].bias)

    def forward(self, action_logits_base, reason_logits_base, predicate_context=None, apply_temperature=True):
        # Backward-compatible guard: old callers may pass apply_temperature as the third positional argument.
        if isinstance(predicate_context, bool):
            apply_temperature = predicate_context
            predicate_context = None
        base_out = super().forward(action_logits_base, reason_logits_base, apply_temperature=apply_temperature)
        if predicate_context is None:
            delta = torch.zeros_like(base_out["logits_base"])
        else:
            delta = self.threshold_delta_max * torch.tanh(self.predicate_to_delta(predicate_context.float()))
        theta = base_out["threshold_logit"].view(1, -1) + delta
        base = base_out["logits_base"]
        deploy = base - theta
        temperature = base_out["temperature"]
        calibrated = deploy / temperature.view(1, -1) if apply_temperature else deploy
        base_out.update({
            "logits_deploy": deploy,
            "logits_calibrated": calibrated,
            "action_logits_deploy": deploy[:, : self.action_dim],
            "reason_logits_deploy": deploy[:, self.action_dim :],
            "action_logits_calibrated": calibrated[:, : self.action_dim],
            "reason_logits_calibrated": calibrated[:, self.action_dim :],
            "threshold_delta": delta,
            "threshold_delta_abs_mean": delta.detach().abs().mean(),
            "threshold_delta_action": delta[:, : self.action_dim],
            "threshold_delta_reason": delta[:, self.action_dim :],
            "predicate_conditioned_threshold_enabled": torch.tensor(predicate_context is not None, device=base.device),
        })
        return base_out
