from __future__ import annotations

import torch
from torch import Tensor, nn


class PACTPredicateAgreement(nn.Module):
    def __init__(self, lambda_max: float = 0.25) -> None:
        super().__init__()
        self.lambda_max = float(lambda_max)
        self.learned_gate = nn.Parameter(torch.zeros(()))

    def forward(self, visual_map: Tensor, predicate_map: Tensor, confidence: Tensor,
                *, bypass: bool = False) -> dict[str, Tensor]:
        visual = visual_map / visual_map.sum(-1, keepdim=True).clamp_min(1e-8)
        predicate = predicate_map / predicate_map.sum(-1, keepdim=True).clamp_min(1e-8)
        agreement = torch.sqrt((visual.clamp_min(0) * predicate.clamp_min(0)).clamp_min(1e-12)).sum(-1)
        confidence = confidence.clamp(0, 1)
        if bypass:
            strength = torch.full_like(agreement, self.lambda_max)
        else:
            strength = self.lambda_max * agreement * confidence * torch.sigmoid(self.learned_gate)
        return {"predicate_visual_agreement": agreement, "predicate_confidence": confidence,
                "predicate_agreement_strength": strength.clamp(0, self.lambda_max),
                "predicate_visual_fallback_rate": ((agreement * confidence) < 1e-4).float().mean()}
