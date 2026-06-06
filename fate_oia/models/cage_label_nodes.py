from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class CAGELabelNodes(nn.Module):
    """First-class action/reason label nodes for BDD-OIA.

    Action and reason labels are represented as category nodes before evidence
    retrieval and dynamic transport. Type embeddings keep action/reason roles
    explicit without hard-coding logits.
    """

    def __init__(self, action_dim: int = 4, reason_dim: int = 21, hidden_dim: int = 256):
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.num_labels = action_dim + reason_dim
        self.hidden_dim = hidden_dim
        self.label_embeddings = nn.Parameter(torch.randn(self.num_labels, hidden_dim) * 0.02)
        self.type_embeddings = nn.Embedding(2, hidden_dim)
        label_type_ids = torch.cat([torch.zeros(action_dim, dtype=torch.long), torch.ones(reason_dim, dtype=torch.long)])
        self.register_buffer("label_type_ids", label_type_ids)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch_size: int | None = None) -> Dict[str, torch.Tensor]:
        queries = self.norm(self.label_embeddings + self.type_embeddings(self.label_type_ids))
        out = {"label_queries": queries, "label_type_ids": self.label_type_ids}
        if batch_size is not None:
            out["batched_label_queries"] = queries.unsqueeze(0).expand(batch_size, -1, -1)
        return out
