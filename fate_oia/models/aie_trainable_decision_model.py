from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class AIETrainableDecisionModel(nn.Module):
    """Trainable decision boundary over a frozen direct-image AIE predictor.

    The visual/reason representation stays frozen. Only four action evidence
    scales and 25 deployment boundaries receive gradients.
    """

    def __init__(
        self,
        base_model: nn.Module,
        *,
        reason_scale: float = 0.6,
        reason_action_scale: float = 0.0,
        action_dim: int = 4,
        reason_dim: int = 21,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.reason_scale = float(reason_scale)
        self.reason_action_scale = float(reason_action_scale)

        for parameter in self.base_model.parameters():
            parameter.requires_grad_(False)
        self.base_model.eval()

        # Conservative initialization keeps evidence auxiliary until gradients
        # demonstrate that an action benefits from a stronger route.
        initial_scale_raw = float(torch.logit(torch.tensor(0.25)))
        self.action_scale_raw = nn.Parameter(torch.full((self.action_dim,), initial_scale_raw))
        initial_reason_scale = torch.full((self.reason_dim,), float(reason_scale)).clamp(1e-6, 1.0 - 1e-6)
        self.reason_scale_raw = nn.Parameter(torch.logit(initial_reason_scale))

        lower = torch.cat((torch.full((self.action_dim,), 0.05), torch.full((self.reason_dim,), 0.01)))
        upper = torch.cat((torch.full((self.action_dim,), 0.95), torch.full((self.reason_dim,), 0.90)))
        initial = torch.full((self.action_dim + self.reason_dim,), 0.5)
        normalized = ((initial - lower) / (upper - lower)).clamp(1e-6, 1.0 - 1e-6)
        self.threshold_raw = nn.Parameter(torch.logit(normalized))
        self.register_buffer("threshold_lower", lower)
        self.register_buffer("threshold_upper", upper)

    @property
    def action_scales(self) -> Tensor:
        return torch.sigmoid(self.action_scale_raw)

    @property
    def reason_scales(self) -> Tensor:
        return torch.sigmoid(self.reason_scale_raw)

    @property
    def threshold_prob(self) -> Tensor:
        unit = torch.sigmoid(self.threshold_raw)
        return self.threshold_lower + (self.threshold_upper - self.threshold_lower) * unit

    def train(self, mode: bool = True) -> "AIETrainableDecisionModel":
        super().train(mode)
        self.base_model.eval()
        return self

    def forward(self, images: Tensor) -> dict[str, Any]:
        with torch.no_grad():
            field = self.base_model.encode_images(images)
        output = self.base_model.decode_from_field(
            field,
            action_scale=self.action_scales,
            reason_scale=self.reason_scales,
            reason_action_scale=self.reason_action_scale,
        )
        final_logits = torch.cat((output["action_logits_final"], output["reason_logits_final"]), dim=-1)
        threshold_prob = self.threshold_prob.to(final_logits)
        threshold_logit = torch.logit(threshold_prob)
        decision_logits = final_logits - threshold_logit.view(1, -1)
        return {
            **output,
            "action_logits_decision": decision_logits[:, : self.action_dim],
            "reason_logits_decision": decision_logits[:, self.action_dim :],
            "action_scales": self.action_scales,
            "reason_scales": self.reason_scales,
            "threshold_prob": threshold_prob,
            "threshold_logit": threshold_logit,
        }
