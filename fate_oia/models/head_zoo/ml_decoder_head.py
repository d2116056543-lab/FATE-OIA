from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.head_zoo.base import BaseOIAHead, with_common_fields


class MLDecoderHead(BaseOIAHead):
    """Small-group ML-Decoder-style head for 25 BDD-OIA labels."""

    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, groups: int = 8, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.num_labels = self.action_dim + self.reason_dim
        self.groups = int(groups)
        self.group_queries = nn.Parameter(torch.empty(self.groups, dim))
        nn.init.trunc_normal_(self.group_queries, std=0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.label_group_logits = nn.Parameter(torch.zeros(self.num_labels, self.groups))
        self.label_proj = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        b = tokens.shape[0]
        q = self.group_queries.unsqueeze(0).expand(b, -1, -1)
        group_tokens, attn = self.attn(q, tokens, tokens, need_weights=True, average_attn_weights=True)
        group_tokens = self.norm(group_tokens)
        weights = torch.softmax(self.label_group_logits, dim=1)
        label_tokens = torch.einsum("lg,bgd->bld", weights, group_tokens)
        logits = self.label_proj(label_tokens).squeeze(-1)
        label_attn = torch.einsum("lg,bgn->bln", weights, attn)
        return with_common_fields(logits, self.action_dim, label_tokens=label_tokens, attention=label_attn, aux_losses={})
