from __future__ import annotations

import torch
from torch import nn


class EaglePUReasonReliability(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21) -> None:
        super().__init__()
        self.head = nn.Sequential(nn.Linear(dim + 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.reason_logit = nn.Linear(dim, 1)
        self.reason_dim = reason_dim

    def forward(self, reason_nodes: torch.Tensor, reason_evidence: torch.Tensor, graph_support: torch.Tensor, evidence_confidence: torch.Tensor, logit_margin: torch.Tensor) -> dict[str, torch.Tensor]:
        aux = torch.stack([graph_support, evidence_confidence, logit_margin], dim=-1).to(reason_nodes.dtype)
        reliability = torch.sigmoid(self.head(torch.cat([reason_nodes, aux], dim=-1)).squeeze(-1))
        logits = self.reason_logit(reason_nodes).squeeze(-1)
        return {"reason_logits": logits, "reason_reliability": reliability}
