from __future__ import annotations

import torch
from torch import nn

from .eagle_pu_sparse_ops import entmax15_bisect


class EaglePULabelDecisionTrunk(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, num_layers: int = 3) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.num_labels = action_dim + reason_dim
        self.layer_router = nn.Parameter(torch.zeros(self.num_labels, num_layers))
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.temperature = nn.Parameter(torch.ones(self.num_labels))
        self.label_self_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.action_visual_head = nn.Linear(dim, 1)
        self.reason_head = nn.Linear(dim, 1)
        self.reason_to_action = nn.Sequential(nn.Linear(reason_dim, dim), nn.GELU(), nn.Linear(dim, action_dim))
        self.gate_head = nn.Linear(dim, action_dim)

    def forward(self, patch_tokens_by_layer: torch.Tensor, label_queries: torch.Tensor, state_tokens: torch.Tensor | None = None) -> dict[str, torch.Tensor | dict[str, float]]:
        b, s, n, d = patch_tokens_by_layer.shape
        layer_weights = torch.softmax(self.layer_router, dim=-1)
        tokens = torch.einsum("ls,bsnd->blnd", layer_weights, patch_tokens_by_layer)
        if state_tokens is not None:
            state_summary = state_tokens.mean(1).unsqueeze(1).unsqueeze(2)
            tokens = tokens + state_summary
        q = self.query_proj(label_queries).unsqueeze(0).expand(b, -1, -1)
        k = self.key_proj(tokens)
        v = self.value_proj(tokens)
        tau = self.temperature.clamp(0.25, 4.0).view(1, -1, 1)
        scores = torch.einsum("bld,blnd->bln", q, k) / (d ** 0.5) / tau
        attn = entmax15_bisect(scores, dim=-1)
        label_evidence = torch.einsum("bln,blnd->bld", attn, v)
        label_nodes, _ = self.label_self_attn(label_evidence, label_evidence, label_evidence)
        label_nodes = label_nodes + label_evidence
        action_nodes = label_nodes[:, : self.action_dim]
        reason_nodes = label_nodes[:, self.action_dim :]
        action_visual_logits = self.action_visual_head(action_nodes).squeeze(-1)
        reason_logits_direct = self.reason_head(reason_nodes).squeeze(-1)
        action_reason_logits = self.reason_to_action(torch.sigmoid(reason_logits_direct))
        gate = torch.sigmoid(self.gate_head(action_nodes.mean(1))).clamp(0.10, 0.90)
        action_logits_direct = gate * action_visual_logits + (1.0 - gate) * action_reason_logits
        entropy = (-(attn.clamp_min(1e-8).log() * attn).sum(-1)).mean()
        support = (attn > 1e-4).float().sum(-1).mean()
        return {
            "label_evidence": label_evidence,
            "label_nodes": label_nodes,
            "label_attention": attn,
            "label_layer_weights": layer_weights,
            "action_visual_logits": action_visual_logits,
            "action_reason_logits": action_reason_logits,
            "action_logits_direct": action_logits_direct,
            "reason_logits_direct": reason_logits_direct,
            "safe_action_gate": gate,
            "attention_stats": {"label_attention_entropy": float(entropy.detach().cpu()), "label_support_size": float(support.detach().cpu())},
        }
