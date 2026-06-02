from __future__ import annotations

import torch
from torch import nn


class _BaseEvidenceExpert(nn.Module):
    source_type = "base"

    def __init__(self, dim: int = 384, geom_dim: int = 8) -> None:
        super().__init__()
        self.geom_proj = nn.Sequential(nn.Linear(geom_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.query_proj = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1)

    def forward(self, reason_tokens: torch.Tensor, tokens: torch.Tensor, bag_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b, r, d = reason_tokens.shape
        global_token = tokens.mean(1).unsqueeze(1).expand(-1, r, -1)
        if bag_features is None:
            geom = torch.zeros(b, r, d, device=tokens.device, dtype=tokens.dtype)
            bag_count = torch.zeros(b, device=tokens.device)
        else:
            bag_emb = self.geom_proj(bag_features.to(tokens.device, tokens.dtype))
            geom = bag_emb.mean(1).unsqueeze(1).expand(-1, r, -1)
            bag_count = (bag_features.abs().sum(-1) > 0).sum(-1).float()
        evidence = torch.tanh(self.query_proj(reason_tokens) + global_token + geom)
        score = self.score(evidence).squeeze(-1)
        reliability = torch.sigmoid(score)
        return {"evidence_tokens": evidence, "evidence_scores": score, "evidence_reliability": reliability, "bag_count": bag_count}


class ObjectEvidenceExpert(_BaseEvidenceExpert):
    source_type = "object"


class LaneEvidenceExpert(_BaseEvidenceExpert):
    source_type = "lane"


class DrivableEvidenceExpert(_BaseEvidenceExpert):
    source_type = "drivable"


class TrafficControlEvidenceExpert(_BaseEvidenceExpert):
    source_type = "traffic_control"


class GlobalContextEvidenceExpert(_BaseEvidenceExpert):
    source_type = "global_context"
