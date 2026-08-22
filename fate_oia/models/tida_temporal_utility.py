from __future__ import annotations

import torch
from torch import nn


class TIDAConditionalTemporalUtility(nn.Module):
    """Allocate bounded per-target temporal budget without label inputs."""

    def __init__(self, max_budget: float, min_budget: float = 0.0) -> None:
        super().__init__()
        self.max_budget = float(max_budget)
        self.min_budget = float(min_budget)
        if not 0.0 <= self.min_budget <= self.max_budget <= 1.0:
            raise ValueError("utility budgets must satisfy 0 <= min <= max <= 1")

    def forward(
        self,
        image_logits: torch.Tensor,
        motion_salience: torch.Tensor,
        transition_consistency: torch.Tensor,
        compatibility: torch.Tensor,
        history_available: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        shape = image_logits.shape
        for name, value in (
            ("motion_salience", motion_salience),
            ("transition_consistency", transition_consistency),
            ("compatibility", compatibility),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must match image logits shape")
        if history_available.shape != shape[:1]:
            raise ValueError("history_available must be [B]")

        probability = image_logits.sigmoid()
        uncertainty = 4.0 * probability * (1.0 - probability)
        motion_weight = motion_salience.clamp_min(0.0) / (1.0 + motion_salience.clamp_min(0.0))
        consistency_weight = transition_consistency.clamp(0.0, 1.0)
        compatibility_weight = 0.5 + 0.5 * compatibility.sigmoid()
        available = history_available[:, None].to(image_logits.dtype)
        need = available * uncertainty * motion_weight * consistency_weight * compatibility_weight
        budget = available * (
            self.min_budget + (self.max_budget - self.min_budget) * need.clamp(0.0, 1.0)
        )
        return {
            "budget": budget,
            "uncertainty": uncertainty,
            "motion_weight": motion_weight,
            "consistency_weight": consistency_weight,
            "compatibility_weight": compatibility_weight,
            "need": need,
            "budget_saturation_rate": (budget >= self.max_budget - 1e-6).to(image_logits.dtype).mean(),
        }
