from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .acpr_sparse_ops import entmax15_bisect


class _LabelSelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_attention = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm_feed_forward = nn.LayerNorm(dim)

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(nodes, nodes, nodes, need_weights=False)
        nodes = self.norm_attention(nodes + attended)
        return self.norm_feed_forward(nodes + self.feed_forward(nodes))


class MOSAICSparseLabelDecoder(nn.Module):
    def __init__(
        self,
        num_labels: int,
        *,
        dim: int = 384,
        decoder_layers: int = 2,
        self_attention_heads: int = 4,
        highres_topk: int = 256,
        midres_topk: int = 128,
        mask_fallback_floor: float = 0.02,
    ) -> None:
        super().__init__()
        for value, name in (
            (num_labels, "num_labels"),
            (dim, "dim"),
            (decoder_layers, "decoder_layers"),
            (self_attention_heads, "self_attention_heads"),
            (highres_topk, "highres_topk"),
            (midres_topk, "midres_topk"),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if dim % self_attention_heads:
            raise ValueError("decoder dim must be divisible by self-attention heads")
        if highres_topk > 45 * 80 or midres_topk > 23 * 40:
            raise ValueError("sparse retrieval budget exceeds available visual tokens")
        if not 0.0 < mask_fallback_floor <= 1.0:
            raise ValueError("mask_fallback_floor must be in (0,1]")

        self.num_labels = num_labels
        self.dim = dim
        self.highres_topk = highres_topk
        self.midres_topk = midres_topk
        self.mask_fallback_floor = float(mask_fallback_floor)
        self.label_queries = nn.Parameter(torch.randn(num_labels, dim) * 0.02)
        self.context_attention = nn.MultiheadAttention(dim, self_attention_heads, batch_first=True)
        self.context_norm = nn.LayerNorm(dim)
        self.high_key = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.high_value = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.mid_key = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.mid_value = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.retrieval_norm = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList(
            [_LabelSelfAttentionBlock(dim, self_attention_heads) for _ in range(decoder_layers)]
        )
        self.final_norm = nn.LayerNorm(dim)
        self.classifier_weight = nn.Parameter(torch.empty(num_labels, dim))
        self.classifier_bias = nn.Parameter(torch.zeros(num_labels))
        nn.init.xavier_uniform_(self.classifier_weight)

    @staticmethod
    def _gather_tokens(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        batch_indices = torch.arange(tokens.shape[0], device=tokens.device).view(-1, 1, 1)
        return tokens[batch_indices, indices]

    def _masked_scores(
        self,
        scores: torch.Tensor,
        masks: torch.Tensor | None,
        spatial_size: tuple[int, int],
    ) -> torch.Tensor:
        if masks is None:
            return scores
        if masks.ndim != 4 or masks.shape[:2] != scores.shape[:2]:
            raise ValueError("label retrieval masks must have shape [B,L,H,W]")
        if tuple(masks.shape[-2:]) != spatial_size:
            masks = F.interpolate(masks, size=spatial_size, mode="bilinear", align_corners=False)
        soft_constraint = self.mask_fallback_floor + (1.0 - self.mask_fallback_floor) * masks.clamp(0.0, 1.0)
        return scores + soft_constraint.flatten(2).log()

    def forward(
        self,
        pyramid: dict[str, torch.Tensor],
        *,
        query_seed: torch.Tensor | None = None,
        highres_masks: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if not isinstance(pyramid, dict) or not {"F_hi", "F_mid", "F_ctx"} <= set(pyramid):
            raise ValueError("sparse label decoder requires F_hi/F_mid/F_ctx")
        high, middle, context = pyramid["F_hi"], pyramid["F_mid"], pyramid["F_ctx"]
        batch_size = high.shape[0]
        if (
            tuple(high.shape) != (batch_size, self.dim, 45, 80)
            or tuple(middle.shape) != (batch_size, self.dim, 23, 40)
            or tuple(context.shape) != (batch_size, self.dim, 12, 20)
        ):
            raise ValueError("sparse label decoder pyramid shape contract is invalid")

        queries = self.label_queries.unsqueeze(0).expand(batch_size, -1, -1)
        if query_seed is not None:
            if tuple(query_seed.shape) != (batch_size, self.num_labels, self.dim):
                raise ValueError("query_seed has an invalid shape")
            queries = queries + query_seed
        context_tokens = context.flatten(2).transpose(1, 2)
        context_nodes, _ = self.context_attention(queries, context_tokens, context_tokens, need_weights=False)
        nodes = self.context_norm(queries + context_nodes)

        high_keys = F.normalize(self.high_key(high).flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        mid_keys = F.normalize(self.mid_key(middle).flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        normalized_nodes = F.normalize(nodes, dim=-1, eps=1e-6)
        high_scores = torch.einsum("bld,bnd->bln", normalized_nodes, high_keys) / math.sqrt(self.dim)
        mid_scores = torch.einsum("bld,bnd->bln", normalized_nodes, mid_keys) / math.sqrt(self.dim)
        high_scores = self._masked_scores(high_scores, highres_masks, (45, 80))
        mid_scores = self._masked_scores(mid_scores, highres_masks, (23, 40))
        high_values, high_indices = high_scores.topk(self.highres_topk, dim=-1)
        mid_values, mid_indices = mid_scores.topk(self.midres_topk, dim=-1)

        high_visual_values = self.high_value(high).flatten(2).transpose(1, 2)
        mid_visual_values = self.mid_value(middle).flatten(2).transpose(1, 2)
        gathered_high = self._gather_tokens(high_visual_values, high_indices)
        gathered_mid = self._gather_tokens(mid_visual_values, mid_indices)
        gathered = torch.cat((gathered_high, gathered_mid), dim=2)
        retrieval_scores = torch.cat((high_values, mid_values), dim=-1)
        retrieval_attention = entmax15_bisect(retrieval_scores, dim=-1)
        retrieved = torch.einsum("blk,blkd->bld", retrieval_attention, gathered)
        nodes = self.retrieval_norm(nodes + retrieved)
        for block in self.blocks:
            nodes = block(nodes)
        nodes = self.final_norm(nodes)
        logits = torch.einsum("bld,ld->bl", nodes, self.classifier_weight) + self.classifier_bias
        return {
            "label_logits": logits,
            "label_nodes": nodes,
            "highres_indices": high_indices,
            "midres_indices": mid_indices,
            "retrieval_attention": retrieval_attention,
            "decoder_stats": {
                "retrieval_support_mean": (retrieval_attention > 1e-5).float().sum(-1).mean().detach(),
            },
        }
