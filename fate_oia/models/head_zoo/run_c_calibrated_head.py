from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.head_zoo.base import BaseOIAHead, split_logits
from fate_oia.models.head_zoo.run_c_compatible_head import RunCCompatibleHead


class RunCCalibratedHead(BaseOIAHead):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, dropout: float = 0.1) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.base = RunCCompatibleHead(dim=dim, action_dim=action_dim, reason_dim=reason_dim, dropout=dropout)
        self.bias = nn.Parameter(torch.zeros(action_dim + reason_dim))
        self.log_temp = nn.Parameter(torch.zeros(action_dim + reason_dim))

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        out = dict(self.base(tokens, labels=labels))
        raw = out["logits"]
        calibrated = raw / torch.exp(self.log_temp).clamp_min(1e-4) + self.bias
        action_logits, reason_logits = split_logits(calibrated, self.action_dim)
        out["raw_logits"] = raw
        out["raw_action_logits"], out["raw_reason_logits"] = split_logits(raw, self.action_dim)
        out["logits"] = calibrated
        out["action_logits"] = action_logits
        out["reason_logits"] = reason_logits
        return out
