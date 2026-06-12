from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from fate_oia.models.cast_sparse_ops import entmax15_bisect


class CastLabelEvidence(nn.Module):
    def __init__(self, dim: int = 384, num_labels: int = 25, selected_layers: int = 3, num_heads: int = 4):
        super().__init__()
        self.dim = int(dim)
        self.num_labels = int(num_labels)
        self.selected_layers = int(selected_layers)
        self.num_heads = int(num_heads)
        self.layer_router = nn.Parameter(torch.zeros(num_labels, selected_layers))
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(num_labels))

    def forward(self, label_queries: torch.Tensor, patch_tokens_by_layer: torch.Tensor, ego_features: torch.Tensor) -> dict:
        # patch_tokens_by_layer: [B,S,N,D]
        b, s, n, d = patch_tokens_by_layer.shape
        if s != self.selected_layers:
            raise ValueError(f"selected layer mismatch: expected {self.selected_layers}, got {s}")
        layer_w = torch.softmax(self.layer_router, dim=-1)
        mixed = torch.einsum("ls,bsnd->blnd", layer_w, patch_tokens_by_layer)
        q = self.query_proj(label_queries).view(1, self.num_labels, 1, d)
        k = self.key_proj(mixed)
        v = self.value_proj(mixed)
        tau = self.temperature.clamp_min(0.05).view(1, self.num_labels, 1)
        logits = (q * k).sum(-1) / math.sqrt(d) / tau
        attn = entmax15_bisect(logits, dim=-1)
        evidence = torch.einsum("bln,blnd->bld", attn, v)
        entropy = -(attn.clamp_min(1e-9) * attn.clamp_min(1e-9).log()).sum(-1)
        support = (attn > 1e-7).float().sum(-1)
        ego = ego_features.to(attn.device, attn.dtype)
        stats = {
            "support_size_mean": float(support.mean().detach().cpu()),
            "entropy_mean": float(entropy.mean().detach().cpu()),
            "left_corridor_mass": float((attn * ego[:, 5].view(1, 1, n)).sum(-1).mean().detach().cpu()),
            "right_corridor_mass": float((attn * ego[:, 6].view(1, 1, n)).sum(-1).mean().detach().cpu()),
            "front_center_mass": float((attn * ego[:, 4].view(1, 1, n)).sum(-1).mean().detach().cpu()),
            "upper_region_mass": float((attn * ego[:, 7].view(1, 1, n)).sum(-1).mean().detach().cpu()),
        }
        return {
            "label_evidence": evidence,
            "label_attention": attn,
            "label_layer_weights": layer_w,
            "attention_stats": stats,
        }
