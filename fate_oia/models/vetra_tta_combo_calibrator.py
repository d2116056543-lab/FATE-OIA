from __future__ import annotations

import torch
from torch import nn


class VetraTTAComboCalibrator(nn.Module):
    """Train-only combo calibrator for original/flip action logits."""

    def __init__(
        self,
        mean: torch.Tensor,
        scale: torch.Tensor,
        coefficient: torch.Tensor,
        intercept: torch.Tensor,
        class_codes: torch.Tensor,
        thresholds: torch.Tensor,
        original_weight: float = 0.75,
    ) -> None:
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("scale", scale.float().clamp_min(1e-6))
        self.register_buffer("coefficient", coefficient.float())
        self.register_buffer("intercept", intercept.float())
        self.register_buffer("class_codes", class_codes.long())
        self.register_buffer("thresholds", thresholds.float())
        self.original_weight = float(original_weight)

    def forward(self, original_logits: torch.Tensor, flipped_logits_remapped: torch.Tensor) -> dict[str, torch.Tensor]:
        mixed = self.original_weight * original_logits + (1.0 - self.original_weight) * flipped_logits_remapped
        standardized = (mixed - self.mean) / self.scale
        combo_logits = standardized @ self.coefficient.t() + self.intercept
        combo_probs = combo_logits.softmax(dim=-1)
        membership = ((self.class_codes[:, None] & (1 << torch.arange(4, device=combo_logits.device))) > 0).float()
        action_probs = combo_probs @ membership
        action_logits = torch.logit(action_probs.clamp(1e-6, 1.0 - 1e-6))
        deploy_logits = action_logits - torch.logit(self.thresholds.clamp(1e-6, 1.0 - 1e-6))
        return {
            "mixed_action_logits": mixed,
            "combo_logits": combo_logits,
            "combo_probs": combo_probs,
            "action_probs": action_probs,
            "action_logits": action_logits,
            "action_deploy_logits": deploy_logits,
        }


def remap_horizontal_flip_actions(logits: torch.Tensor) -> torch.Tensor:
    result = logits.clone()
    result[:, 2], result[:, 3] = logits[:, 3], logits[:, 2]
    return result
