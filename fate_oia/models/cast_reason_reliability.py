from __future__ import annotations

import torch
from torch import nn


class CastReasonReliability(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21):
        super().__init__()
        self.reason_dim = int(reason_dim)
        self.reason_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.reason_head.bias)
        self.reliability_head = nn.Sequential(nn.Linear(dim + 3, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, reason_nodes, reason_evidence, graph_support, evidence_confidence, logit_margin):
        fused = reason_nodes + reason_evidence
        reason_logits = self.reason_head(fused).squeeze(-1)
        aux = torch.stack([graph_support, evidence_confidence, logit_margin], dim=-1).to(fused.dtype)
        reliability = torch.sigmoid(self.reliability_head(torch.cat([fused, aux], dim=-1)).squeeze(-1))
        return {"reason_logits": reason_logits, "reason_reliability": reliability}
