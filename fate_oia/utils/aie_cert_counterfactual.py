from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def target_signed_margin(logits: Tensor, target: Tensor) -> Tensor:
    return (2.0 * target - 1.0) * logits


@dataclass
class AIECertCounterfactualConfig:
    topk: int = 64
    min_valid_controls: int = 3
    max_overlap: float = 0.20
    reliability_tau: float = 0.25


class AIECertCounterfactualEngine:
    """Build robust selected-vs-four-control certificates without re-encoding DINO."""
    def __init__(self, config: AIECertCounterfactualConfig | None = None):
        self.config = config or AIECertCounterfactualConfig()

    def summarize(self, original_margin: Tensor, selected_margin: Tensor, control_margins: Tensor,
                  control_valid: Tensor) -> dict[str, Tensor]:
        selected_drop = original_margin - selected_margin
        control_drop = original_margin[..., None] - control_margins
        valid_count = control_valid.sum(-1)
        masked = torch.where(control_valid, control_drop, torch.zeros_like(control_drop))
        mean = masked.sum(-1) / valid_count.clamp_min(1)
        variance = torch.where(control_valid, (control_drop - mean[..., None]).square(), torch.zeros_like(control_drop))
        # A zero-variance control set is valid, but sqrt'(0) is infinite and
        # poisons the evidence owner during the certificate backward pass.
        control_variance = variance.sum(-1) / valid_count.clamp_min(1)
        std = torch.sqrt(control_variance.clamp_min(1e-6))
        valid = valid_count >= self.config.min_valid_controls
        certificate = selected_drop - (mean + std)
        reliability = torch.exp(-std / self.config.reliability_tau)
        return {"selected_drop": selected_drop, "control_drops": control_drop, "control_mean": mean,
                "control_std": std, "certificate": certificate, "reliability": reliability,
                "valid_mask": valid, "per_control_validity": control_valid}
