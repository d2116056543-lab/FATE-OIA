from __future__ import annotations

import torch
from torch import nn


class ACPRCalibrationHead(nn.Module):
    def __init__(self, num_labels: int = 25) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(num_labels))
        self.bias = nn.Parameter(torch.zeros(num_labels))

    def forward(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        temperature = torch.exp(self.log_temperature).clamp(0.5, 3.0)
        bias = self.bias.clamp(-2.0, 2.0)
        return {"calibrated_logits": logits / temperature.view(1, -1) + bias.view(1, -1), "temperature": temperature, "calibration_bias": bias}
