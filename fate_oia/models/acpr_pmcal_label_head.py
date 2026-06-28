from __future__ import annotations

import torch
from torch import nn

from .acpr_sparse_ops import entmax15_bisect


class ACPRPMCalLabelHead(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.num_labels = self.action_dim + self.reason_dim
        self.label_queries = nn.Parameter(torch.randn(self.num_labels, dim) * 0.02)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.label_self_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.reason_head = nn.Linear(dim, 1)

    def forward(self, patch_tokens_by_layer: torch.Tensor, predicate_tokens: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        patch = patch_tokens_by_layer.mean(1)
        b, n, d = patch.shape
        q = self.query_proj(self.label_queries).view(1, self.num_labels, 1, d)
        k = self.key_proj(patch).view(b, 1, n, d)
        v = self.value_proj(patch)
        score = (q * k).sum(-1) / (d ** 0.5)
        attn = entmax15_bisect(score, dim=-1)
        label_nodes = torch.einsum("bln,bnd->bld", attn, v)
        label_nodes = label_nodes + self.label_self_attn(label_nodes, label_nodes, label_nodes, need_weights=False)[0]
        reason_nodes = label_nodes[:, self.action_dim :]
        reason_logits_visual = self.reason_head(reason_nodes).squeeze(-1)
        return {
            "label_nodes": label_nodes,
            "label_attention": attn,
            "action_nodes": label_nodes[:, : self.action_dim],
            "reason_nodes": reason_nodes,
            "reason_logits_visual": reason_logits_visual,
        }
