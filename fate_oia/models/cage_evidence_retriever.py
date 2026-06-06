from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn


class CAGEEvidenceRetriever(nn.Module):
    """Label-specific evidence retriever over image tokens.

    Each action/reason label owns a query and receives its own attention map over
    visual tokens. This avoids the generic factor-slot failure seen in DIVA.
    """

    def __init__(self, hidden_dim: int, num_labels: int = 25, num_heads: int = 4, topk: int = 8):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_labels <= 0:
            raise ValueError("num_labels must be positive")
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.num_heads = num_heads
        self.topk = topk
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor, label_queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        if tokens.dim() != 3:
            raise ValueError(f"tokens must be [B,N,D], got {tuple(tokens.shape)}")
        if label_queries.dim() != 2:
            raise ValueError(f"label_queries must be [L,D], got {tuple(label_queries.shape)}")
        bsz, num_tokens, dim = tokens.shape
        num_labels, q_dim = label_queries.shape
        if dim != self.hidden_dim or q_dim != self.hidden_dim:
            raise ValueError("token/query dim must match hidden_dim")
        if num_labels != self.num_labels:
            raise ValueError(f"expected {self.num_labels} labels, got {num_labels}")

        q = self.query_proj(label_queries).unsqueeze(0).expand(bsz, -1, -1)
        k = self.key_proj(tokens)
        v = self.value_proj(tokens)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(dim))
        evidence_scores = torch.softmax(scores, dim=-1)
        evidence_state = torch.matmul(evidence_scores, v)
        evidence_state = self.norm(self.out_proj(evidence_state) + q)

        k_eff = min(max(int(self.topk), 1), num_tokens)
        topk_scores, topk_indices = torch.topk(evidence_scores, k=k_eff, dim=-1)
        return {
            "evidence_state": evidence_state,
            "evidence_scores": evidence_scores,
            "topk_scores": topk_scores,
            "topk_indices": topk_indices,
            "evidence_confidence": topk_scores.mean(dim=-1),
        }
