from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class AdaptiveCalibrationHead(nn.Module):
    """Global or instance-conditioned calibration for multi-label logits."""

    def __init__(self, dim: int, num_labels: int, mode: str = "none", delta_clip: float = 2.0) -> None:
        super().__init__()
        if mode not in {"none", "global", "instance"}:
            raise ValueError(f"Unsupported adaptive_calibration mode: {mode}")
        self.mode = mode
        self.num_labels = int(num_labels)
        self.delta_clip = float(delta_clip)
        self.global_bias = nn.Parameter(torch.zeros(num_labels))
        self.instance = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_labels))

    def forward(self, logits: torch.Tensor, state_tokens: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.mode == "none":
            delta = torch.zeros_like(logits)
            bias = torch.zeros_like(logits)
            calibrated = logits
        else:
            bias = self.global_bias.view(1, -1).expand_as(logits)
            if self.mode == "instance" and state_tokens is not None:
                delta = self.instance(state_tokens.mean(dim=1)).clamp(-self.delta_clip, self.delta_clip)
            else:
                delta = torch.zeros_like(logits)
            calibrated = logits - bias - delta
        return {
            "calibrated_logits": calibrated,
            "calibration_bias_global": bias,
            "calibration_delta_instance": delta,
            "calibration_mean_abs_delta": delta.detach().abs().mean(),
            "calibration_mean_abs_global_bias": self.global_bias.detach().abs().mean(),
        }

    def calibration_loss(self, calibrated_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(calibrated_logits, targets.float())
