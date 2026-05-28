from __future__ import annotations

import torch
from torch import nn


class SemanticLabelQueries(nn.Module):
    """Learned action/reason query table with light type embeddings."""

    def __init__(self, num_labels: int, dim: int, action_dim: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_labels = int(num_labels)
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.query = nn.Parameter(torch.empty(self.num_labels, self.dim))
        self.type_embed = nn.Embedding(2, self.dim)
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        device = device or self.query.device
        q = self.query.to(device).unsqueeze(0).expand(int(batch_size), -1, -1)
        type_ids = torch.ones(self.num_labels, dtype=torch.long, device=device)
        type_ids[: self.action_dim] = 0
        q = q + self.type_embed(type_ids).unsqueeze(0)
        return self.dropout(self.norm(q))
