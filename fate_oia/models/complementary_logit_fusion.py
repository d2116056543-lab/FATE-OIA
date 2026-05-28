from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ComplementaryLogitFusionAdapter(nn.Module):
    """Per-label cached-logit fusion baseline.

    This intentionally has no hidden network: each reason label learns only a
    RunC/ScoreV2 mix weight, additive bias, and positive temperature.
    """

    def __init__(self, reason_dim: int = 21, init_mix: float = 0.0) -> None:
        super().__init__()
        init_mix = min(max(float(init_mix), 1e-4), 1.0 - 1e-4)
        self.mix_logit = nn.Parameter(torch.full((reason_dim,), math.log(init_mix / (1.0 - init_mix))))
        self.bias = nn.Parameter(torch.zeros(reason_dim))
        self.log_temperature = nn.Parameter(torch.zeros(reason_dim))

    def mix_weight(self) -> torch.Tensor:
        return torch.sigmoid(self.mix_logit)

    def temperature(self) -> torch.Tensor:
        return F.softplus(self.log_temperature) + 1e-4

    def forward(self, run_c_reason_logits: torch.Tensor, score_v2_reason_logits: torch.Tensor) -> torch.Tensor:
        mix = self.mix_weight().view(1, -1)
        temp = self.temperature().view(1, -1)
        fused = (1.0 - mix) * run_c_reason_logits + mix * score_v2_reason_logits
        return fused / temp + self.bias.view(1, -1)

