from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .acpr_sparse_ops import entmax15_bisect


class _CategorySparseCrossLayer(nn.Module):
    def __init__(self, dim: int, highres_topk: int, midres_topk: int) -> None:
        super().__init__()
        self.highres_topk = int(highres_topk)
        self.midres_topk = int(midres_topk)
        self.query = nn.Linear(dim, dim, bias=False)
        self.high_key = nn.Linear(dim, dim, bias=False)
        self.high_value = nn.Linear(dim, dim, bias=False)
        self.mid_key = nn.Linear(dim, dim, bias=False)
        self.mid_value = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    @staticmethod
    def _tokens(feature_map: torch.Tensor) -> torch.Tensor:
        return feature_map.flatten(2).transpose(1, 2)

    def _sparse_read(
        self,
        nodes: torch.Tensor,
        feature_map: torch.Tensor,
        key_proj: nn.Linear,
        value_proj: nn.Linear,
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokens(feature_map)
        query = F.normalize(self.query(nodes), dim=-1, eps=1e-6)
        keys = F.normalize(key_proj(tokens), dim=-1, eps=1e-6)
        logits = torch.einsum("bqd,bnd->bqn", query, keys) / math.sqrt(nodes.shape[-1])
        count = min(int(topk), tokens.shape[1])
        top_logits, top_indices = logits.topk(count, dim=-1)
        sparse_weights = entmax15_bisect(top_logits, dim=-1)
        values = value_proj(tokens)
        gathered = values.unsqueeze(1).expand(-1, nodes.shape[1], -1, -1).gather(
            2, top_indices.unsqueeze(-1).expand(-1, -1, -1, values.shape[-1])
        )
        read = torch.einsum("bqk,bqkd->bqd", sparse_weights, gathered)
        # entmax may promote to fp32 inside bf16 autocast. Scatter must inherit
        # the source weights dtype rather than the pre-entmax logits dtype.
        full_attention = sparse_weights.new_zeros(logits.shape)
        full_attention.scatter_(2, top_indices, sparse_weights)
        return read, full_attention

    def forward(self, nodes: torch.Tensor, pyramid: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        high_read, high_attention = self._sparse_read(
            nodes, pyramid["F_hi"], self.high_key, self.high_value, self.highres_topk
        )
        mid_read, _ = self._sparse_read(nodes, pyramid["F_mid"], self.mid_key, self.mid_value, self.midres_topk)
        updated = self.norm(nodes + high_read + mid_read)
        return self.norm(updated + self.ffn(updated)), high_attention


class MOSAICICDORActionDecoder(nn.Module):
    """Action-owned visual decoder with no factor, reason, or state inputs."""

    def __init__(
        self,
        *,
        dim: int = 384,
        action_count: int = 4,
        decoder_layers: int = 2,
        self_attention_heads: int = 4,
        highres_topk: int = 256,
        midres_topk: int = 128,
    ) -> None:
        super().__init__()
        if action_count != 4:
            raise ValueError("IC-DOR action decoder requires four BDD-OIA actions")
        if decoder_layers != 2 or self_attention_heads != 4:
            raise ValueError("IC-DOR action decoder requires two sparse layers and four self-attention heads")
        self.dim = int(dim)
        self.action_count = int(action_count)
        self.action_queries = nn.Parameter(torch.randn(action_count, dim) * 0.02)
        self.cross_layers = nn.ModuleList(
            [_CategorySparseCrossLayer(dim, highres_topk, midres_topk) for _ in range(decoder_layers)]
        )
        self.query_self_attention = nn.MultiheadAttention(dim, self_attention_heads, batch_first=True)
        self.self_norm = nn.LayerNorm(dim)
        self.logit_head = nn.Linear(dim, 1)

    def forward(self, pyramid: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        required = {"F_hi", "F_mid", "F_ctx"}
        if not isinstance(pyramid, dict) or not required <= set(pyramid):
            raise ValueError("IC-DOR action decoder requires F_hi/F_mid/F_ctx action pyramid")
        high = pyramid["F_hi"]
        if high.ndim != 4 or high.shape[1] != self.dim or tuple(high.shape[-2:]) != (45, 80):
            raise ValueError("IC-DOR action decoder expects F_hi [B,D,45,80]")
        batch_size = high.shape[0]
        nodes = self.action_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attention = None
        for layer in self.cross_layers:
            nodes, attention = layer(nodes, pyramid)
        self_nodes, _ = self.query_self_attention(nodes, nodes, nodes, need_weights=False)
        nodes = self.self_norm(nodes + self_nodes)
        if attention is None:
            raise RuntimeError("IC-DOR action decoder did not execute category sparse attention")
        return {
            "action_visual_nodes": nodes,
            "action_visual_attention": attention.reshape(batch_size, self.action_count, 45, 80),
            "action_visual_logits": self.logit_head(nodes).squeeze(-1),
            "action_queries": nodes,
        }
