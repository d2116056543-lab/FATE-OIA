from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch import nn


class SURESparseRelationAttention(nn.Module):
    def __init__(self, dim: int, label_count: int = 25, max_edges_per_label: int = 8, max_edges_total: int = 96) -> None:
        super().__init__()
        self.dim = dim
        self.label_count = label_count
        self.max_edges_per_label = max_edges_per_label
        self.max_edges_total = max_edges_total
        self.label_proj = nn.Linear(dim, dim, bias=False)
        self.relation_proj = nn.Linear(dim, dim, bias=False)
        self.context_proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())

    def forward(self, label_tokens: torch.Tensor, relation_tokens: torch.Tensor, relation_type_ids: torch.Tensor | None = None) -> dict[str, Any]:
        q = self.label_proj(label_tokens)
        k = self.relation_proj(relation_tokens)
        scores = torch.matmul(q, k.transpose(1, 2)) / (self.dim**0.5)
        bsz, labels, rels = scores.shape
        per_label_k = min(self.max_edges_per_label, rels)
        vals, idx = scores.topk(per_label_k, dim=-1)
        flat_vals = vals.reshape(bsz, -1)
        flat_idx = idx.reshape(bsz, -1)
        total_k = min(self.max_edges_total, flat_vals.shape[1])
        top_vals, top_pos = flat_vals.topk(total_k, dim=-1)
        selected_rel_idx = torch.gather(flat_idx, 1, top_pos)
        selected_label_idx = top_pos // per_label_k
        rel_gather = relation_tokens.gather(1, selected_rel_idx.unsqueeze(-1).expand(-1, -1, relation_tokens.shape[-1]))
        weights = torch.softmax(top_vals, dim=-1).unsqueeze(-1)
        global_context = (rel_gather * weights).sum(1)
        label_context = global_context.unsqueeze(1).expand(-1, labels, -1)
        label_context = self.context_proj(label_context)
        type_counts: list[dict[str, int]] = []
        if relation_type_ids is not None:
            for b in range(bsz):
                ids = relation_type_ids[b].gather(0, selected_rel_idx[b]).detach().cpu().tolist()
                type_counts.append({str(k): int(v) for k, v in Counter(ids).items()})
        stats = {
            "candidate_edges": int(bsz * labels * rels),
            "selected_edges": int(bsz * total_k),
            "candidate_relations": int(rels),
            "selected_relations_per_batch": int(total_k),
            "type_counts": type_counts,
        }
        return {
            "label_context": label_context,
            "selected_relation_indices": selected_rel_idx,
            "selected_label_indices": selected_label_idx,
            "edge_scores": top_vals,
            "stats": stats,
        }
