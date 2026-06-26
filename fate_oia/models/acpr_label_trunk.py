from __future__ import annotations

import torch
from torch import nn

from .acpr_grounded_evidence_memory import ACPREvidenceMemoryAugmenter
from .acpr_sparse_ops import entmax15_bisect


class ACPRLabelTrunk(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.num_labels = action_dim + reason_dim
        self.label_queries = nn.Parameter(torch.randn(self.num_labels, dim) * 0.02)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.query_proj = nn.Linear(dim, dim)
        self.label_self_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.predicate_cross_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.predicate_gate = nn.Parameter(torch.full((reason_dim,), -2.944))
        self.logit_head = nn.Linear(dim, 1)
        self.reason_to_action = nn.Linear(reason_dim, action_dim)
        hidden = max(dim // 2, 1)
        self.action_visual_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.fusion_gate = nn.Linear(dim * 2, action_dim)
        self.evidence_augmenter = ACPREvidenceMemoryAugmenter(dim=dim, max_delta=0.20, num_heads=4)
        self.evidence_out_proj = self.evidence_augmenter.evidence_out_proj

    def forward(
        self,
        patch_tokens_by_layer: torch.Tensor,
        predicate_tokens: torch.Tensor | None = None,
        evidence_tokens: torch.Tensor | None = None,
        evidence_attention: torch.Tensor | None = None,
        evidence_enabled: bool = False,
    ) -> dict[str, torch.Tensor]:
        patch = patch_tokens_by_layer.mean(1)
        b, n, d = patch.shape
        q = self.query_proj(self.label_queries).view(1, self.num_labels, 1, d)
        k = self.key_proj(patch).view(b, 1, n, d)
        v = self.value_proj(patch)
        score = (q * k).sum(-1) / (d ** 0.5)
        attn = entmax15_bisect(score, dim=-1)
        label_nodes = torch.einsum("bln,bnd->bld", attn, v)
        label_nodes = label_nodes + self.label_self_attn(label_nodes, label_nodes, label_nodes, need_weights=False)[0]
        label_nodes_patch = label_nodes
        if evidence_enabled and evidence_tokens is not None:
            label_nodes, evidence_context, label_evidence_attention = self.evidence_augmenter(label_nodes_patch, evidence_tokens)
            label_evidence_delta = label_nodes - label_nodes_patch
        else:
            evidence_context = torch.zeros_like(label_nodes)
            label_evidence_delta = torch.zeros_like(label_nodes)
            label_evidence_attention = label_nodes.new_zeros(b, self.num_labels, 0)
        if predicate_tokens is not None:
            reason_nodes = label_nodes[:, self.action_dim :]
            pred_delta = self.predicate_cross_attn(reason_nodes, predicate_tokens, predicate_tokens, need_weights=False)[0]
            gate = torch.sigmoid(self.predicate_gate).clamp(max=0.20).view(1, self.reason_dim, 1)
            label_nodes = torch.cat([label_nodes[:, : self.action_dim], reason_nodes + gate * pred_delta], dim=1)
        label_logits = self.logit_head(label_nodes).squeeze(-1)
        reason_logits_visual = label_logits[:, self.action_dim:]
        action_nodes = label_nodes[:, : self.action_dim]
        action_visual_logits = self.action_visual_head(action_nodes).squeeze(-1)
        action_reason_logits = self.reason_to_action(reason_logits_visual)
        gate_in = torch.cat([label_nodes[:, : self.action_dim].mean(1), label_nodes[:, self.action_dim:].mean(1)], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(gate_in)).clamp(0.10, 0.90)
        action_logits_direct = gate * action_visual_logits + (1.0 - gate) * action_reason_logits
        return {
            "label_nodes": label_nodes,
            "label_nodes_patch": label_nodes_patch,
            "label_nodes_evidence_context": evidence_context,
            "label_nodes_evidence_delta": label_evidence_delta,
            "label_attention": attn,
            "label_evidence_attention": label_evidence_attention,
            "action_evidence_attention": label_evidence_attention[:, : self.action_dim],
            "reason_evidence_attention": label_evidence_attention[:, self.action_dim :],
            "label_evidence_delta_norm": label_evidence_delta.norm(dim=-1).mean(),
            "label_evidence_gate_or_scale_stats": torch.tensor(
                [float(label_evidence_delta.norm(dim=-1).mean().detach().cpu())],
                device=label_nodes.device,
                dtype=label_nodes.dtype,
            ),
            "action_visual_logits": action_visual_logits,
            "action_reason_logits": action_reason_logits,
            "reason_logits_visual": reason_logits_visual,
            "action_fusion_gate": gate,
            "action_token_norm_mean": action_nodes.norm(dim=-1).mean(),
            "predicate_conditioning_strength": torch.sigmoid(self.predicate_gate).detach(),
            "action_logits_direct": action_logits_direct,
        }
