from __future__ import annotations

import torch
from torch import Tensor, nn


class DICELicensePredictor(nn.Module):
    def __init__(self, dim: int = 384, hidden: int = 128) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden + 8, hidden), nn.GELU(), nn.Linear(hidden, 2))

    def forward(self, token: Tensor, evidence_map: Tensor, agreement: Tensor, confidence: Tensor,
                base_logits: Tensor, legacy_contribution: Tensor) -> dict[str, Tensor]:
        probs = torch.sigmoid(base_logits).unsqueeze(-1).expand_as(agreement)
        uncertainty = (4 * probs * (1 - probs)).clamp(0, 1)
        maps = evidence_map.clamp_min(1e-8)
        normalizer = torch.log(torch.as_tensor(max(maps.shape[-1], 2), device=maps.device, dtype=maps.dtype))
        entropy = -(maps * maps.log()).sum(-1) / normalizer
        top = maps.topk(min(2, maps.shape[-1]), -1).values
        peak, gap = top[..., 0], top[..., 0] - (top[..., 1] if top.shape[-1] > 1 else 0)
        legacy = legacy_contribution.detach()
        features = torch.stack((entropy, peak, gap, agreement, confidence, probs, uncertainty, legacy), -1)
        logits = self.head(torch.cat((self.token(token), features), -1))
        license_value = torch.sigmoid(logits)
        return {"license_support_hat": license_value[..., 0], "license_counter_hat": license_value[..., 1],
                "license_logits": logits}
