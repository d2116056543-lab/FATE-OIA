from __future__ import annotations

import torch
from torch import nn


class ACPRCalibrationHead(nn.Module):
    def __init__(self, num_labels: int = 25, action_dim: int = 4, reason_dim: int = 21) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        if num_labels != action_dim + reason_dim:
            num_labels = action_dim + reason_dim
        self.log_temperature = nn.Parameter(torch.zeros(num_labels))
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, action_logits_raw: torch.Tensor, reason_logits_raw: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if reason_logits_raw is None:
            logits = action_logits_raw
            action_logits_raw = logits[:, : self.action_dim]
            reason_logits_raw = logits[:, self.action_dim :]
        else:
            logits = torch.cat([action_logits_raw, reason_logits_raw], dim=-1)
        temperature = torch.exp(self.log_temperature).clamp(0.5, 3.0)
        bias = self.bias.clamp(-2.0, 2.0)
        calibrated_logits = logits / temperature.view(1, -1) + bias.view(1, -1)
        return {
            "calibrated_logits": calibrated_logits,
            "action_logits_calibrated": calibrated_logits[:, : self.action_dim],
            "reason_logits_calibrated": calibrated_logits[:, self.action_dim :],
            "temperature": temperature,
            "calibration_bias": bias,
            "temperature_action": temperature[: self.action_dim],
            "temperature_reason": temperature[self.action_dim :],
            "bias_action": bias[: self.action_dim],
            "bias_reason": bias[self.action_dim :],
        }
