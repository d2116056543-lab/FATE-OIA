from __future__ import annotations

import torch
from torch import nn


class SparseRegionAttention(nn.Module):
    """Pure-PyTorch sparse top-k attention fallback for pair/evidence branches."""

    def __init__(self, dim: int = 384, topk: int = 64) -> None:
        super().__init__()
        self.topk = int(topk)
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, queries: torch.Tensor, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        q = self.query(queries)
        k = self.key(tokens)
        v = self.value(tokens)
        scores = torch.matmul(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        k_count = min(self.topk, scores.shape[-1])
        top_scores, top_idx = torch.topk(scores, k_count, dim=-1)
        weights = torch.softmax(top_scores, dim=-1)
        expanded_v = v.unsqueeze(1).expand(-1, queries.shape[1], -1, -1)
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1])
        selected_v = torch.gather(expanded_v, 2, gather_idx)
        pooled = (weights.unsqueeze(-1) * selected_v).sum(dim=2)
        return {"pooled": pooled, "indices": top_idx, "weights": weights}
