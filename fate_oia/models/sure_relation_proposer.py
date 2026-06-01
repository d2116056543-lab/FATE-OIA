from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from fate_oia.utils.sure_edge_types import EDGE_TYPES, edge_type_id


def _infer_grid(num_tokens: int, image_meta: list[dict[str, Any]] | None = None) -> tuple[int, int]:
    if image_meta and image_meta[0]:
        meta = image_meta[0]
        gh = int(meta.get("patch_grid_h", 0) or meta.get("grid_h", 0) or 0)
        gw = int(meta.get("patch_grid_w", 0) or meta.get("grid_w", 0) or 0)
        if gh > 0 and gw > 0 and gh * gw <= num_tokens:
            return gh, gw
    root = int(math.sqrt(num_tokens))
    for h in range(root, 0, -1):
        if num_tokens % h == 0:
            return h, num_tokens // h
    return 1, num_tokens


def _pool_box_tokens(tokens: torch.Tensor, box: dict[str, Any], grid_h: int, grid_w: int, image_w: float = 1280.0, image_h: float = 720.0) -> torch.Tensor:
    x1 = float(box.get("x1", 0.0)) / max(image_w, 1.0)
    x2 = float(box.get("x2", image_w)) / max(image_w, 1.0)
    y1 = float(box.get("y1", 0.0)) / max(image_h, 1.0)
    y2 = float(box.get("y2", image_h)) / max(image_h, 1.0)
    c1 = max(0, min(grid_w - 1, int(math.floor(min(x1, x2) * grid_w))))
    c2 = max(c1 + 1, min(grid_w, int(math.ceil(max(x1, x2) * grid_w))))
    r1 = max(0, min(grid_h - 1, int(math.floor(min(y1, y2) * grid_h))))
    r2 = max(r1 + 1, min(grid_h, int(math.ceil(max(y1, y2) * grid_h))))
    idx = []
    for r in range(r1, r2):
        for c in range(c1, c2):
            flat = r * grid_w + c
            if flat < tokens.shape[0]:
                idx.append(flat)
    if not idx:
        return tokens.mean(0)
    return tokens[torch.tensor(idx, device=tokens.device)].mean(0)


class SURERelationProposer(nn.Module):
    """Build relation candidates from image patch tokens.

    Fair-path candidates use only image tokens. Structured BDD100K records are
    used for GT-scene upper-bound tokens and diagnostics.
    """

    def __init__(self, dim: int, relation_queries: int = 32) -> None:
        super().__init__()
        self.dim = dim
        self.relation_queries = relation_queries
        self.query_embed = nn.Parameter(torch.randn(relation_queries, dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.type_embed = nn.Embedding(len(EDGE_TYPES), dim)
        self.score_head = nn.Linear(dim, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        structured: list[dict[str, Any]] | None = None,
        image_meta: list[dict[str, Any]] | None = None,
        use_gt_scene_upper: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        bsz, _, dim = tokens.shape
        queries = self.query_embed.unsqueeze(0).expand(bsz, -1, -1)
        rel_tokens, attn = self.cross_attn(queries, tokens, tokens, need_weights=True)
        fair_type_ids = torch.full((bsz, self.relation_queries), edge_type_id("patch_context"), dtype=torch.long, device=tokens.device)
        rel_tokens = rel_tokens + self.type_embed(fair_type_ids)

        gt_tokens = self._build_gt_tokens(tokens, structured, image_meta) if structured else torch.zeros_like(rel_tokens)
        output_tokens = gt_tokens if use_gt_scene_upper else rel_tokens
        scores = self.score_head(output_tokens).squeeze(-1)
        type_ids = fair_type_ids
        stats = {
            "candidate_relations": int(output_tokens.shape[1]),
            "gt_scene_upper": bool(use_gt_scene_upper),
            "structured_records": int(len(structured) if structured else 0),
            "mean_relation_score": float(scores.detach().mean().cpu().item()),
        }
        return {"relation_tokens": output_tokens, "relation_scores": scores, "relation_type_ids": type_ids, "attention": attn, "stats": stats}

    def _build_gt_tokens(
        self,
        tokens: torch.Tensor,
        structured: list[dict[str, Any]] | None,
        image_meta: list[dict[str, Any]] | None,
    ) -> torch.Tensor:
        bsz, num_tokens, dim = tokens.shape
        grid_h, grid_w = _infer_grid(num_tokens, image_meta)
        rows = []
        for b in range(bsz):
            rec = structured[b] if structured and b < len(structured) else {}
            pooled: list[torch.Tensor] = []
            for obj in rec.get("objects", [])[: self.relation_queries]:
                box = obj.get("box2d")
                if box:
                    pooled.append(_pool_box_tokens(tokens[b], box, grid_h, grid_w))
            for _lane in rec.get("lanes", [])[: max(0, self.relation_queries - len(pooled))]:
                pooled.append(tokens[b].mean(0))
            if rec.get("drivable") and len(pooled) < self.relation_queries:
                pooled.append(tokens[b, num_tokens // 2 :].mean(0))
            while len(pooled) < self.relation_queries:
                start = (len(pooled) * num_tokens) // self.relation_queries
                end = max(start + 1, ((len(pooled) + 1) * num_tokens) // self.relation_queries)
                pooled.append(tokens[b, start:end].mean(0))
            rows.append(torch.stack(pooled[: self.relation_queries], dim=0))
        return torch.stack(rows, dim=0)
