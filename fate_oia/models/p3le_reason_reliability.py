from __future__ import annotations

import torch
from torch import nn


class ReasonReliabilityHead(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21, tail_indices: tuple[int, ...] = (5, 6, 9, 10, 11, 12, 13, 14)) -> None:
        super().__init__()
        self.reason_dim = int(reason_dim)
        tail = torch.zeros(reason_dim)
        for idx in tail_indices:
            if 0 <= idx < reason_dim:
                tail[idx] = 1.0
        self.register_buffer("tail_prior", tail)
        self.mlp = nn.Sequential(nn.Linear(dim + 5, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def forward(
        self,
        reason_tokens: torch.Tensor,
        reason_logits: torch.Tensor,
        pair_reason_support: torch.Tensor,
        action_agreement: torch.Tensor,
        evidence_score: torch.Tensor,
        epoch: int = 0,
        warmup_epochs: int = 5,
    ) -> dict[str, torch.Tensor]:
        probs = torch.sigmoid(reason_logits)
        uncertainty = 1.0 - (probs - 0.5).abs() * 2.0
        tail = self.tail_prior.to(reason_tokens.device, reason_tokens.dtype).unsqueeze(0).expand(reason_tokens.shape[0], -1)
        action_feat = action_agreement.mean(dim=1, keepdim=True).expand(-1, self.reason_dim)
        features = torch.cat(
            [
                reason_tokens,
                reason_logits.unsqueeze(-1),
                pair_reason_support.unsqueeze(-1),
                uncertainty.unsqueeze(-1),
                action_feat.unsqueeze(-1),
                evidence_score.unsqueeze(-1),
            ],
            dim=-1,
        )
        raw_q = torch.sigmoid(self.mlp(features).squeeze(-1))
        min_q = 0.7 if epoch < warmup_epochs else 0.05
        q = raw_q.clamp(min=min_q, max=0.98)
        return {"reason_reliability": q, "reason_reliability_raw": raw_q, "reason_uncertainty": uncertainty, "tail_prior": tail}
