from __future__ import annotations

import torch
from torch import nn

from .acpr_semantic_evidence_coattention import ACPRSparseEvidenceCoAttention
from .acpr_sparse_ops import entmax15_bisect


class ACPRLabelTrunk(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        *,
        seca_enabled: bool = False,
        seca_num_heads: int = 4,
        seca_max_residual_scale: float = 0.20,
        seca_evidence_grad_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.num_labels = action_dim + reason_dim
        self.seca_enabled = bool(seca_enabled)
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
        self.seca = ACPRSparseEvidenceCoAttention(
            dim=dim,
            action_dim=action_dim,
            reason_dim=reason_dim,
            num_heads=seca_num_heads,
            max_residual_scale=seca_max_residual_scale,
            evidence_grad_scale=seca_evidence_grad_scale,
        )

    def _action_outputs(self, label_nodes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        label_logits = self.logit_head(label_nodes).squeeze(-1)
        reason_logits_visual = label_logits[:, self.action_dim:]
        action_nodes = label_nodes[:, : self.action_dim]
        action_visual_logits = self.action_visual_head(action_nodes).squeeze(-1)
        action_reason_logits = self.reason_to_action(reason_logits_visual)
        gate_in = torch.cat([action_nodes.mean(1), label_nodes[:, self.action_dim:].mean(1)], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(gate_in)).clamp(0.10, 0.90)
        action_logits_direct = gate * action_visual_logits + (1.0 - gate) * action_reason_logits
        return reason_logits_visual, action_visual_logits, action_reason_logits, gate, action_logits_direct

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
        if predicate_tokens is not None:
            reason_nodes = label_nodes[:, self.action_dim :]
            pred_delta, pred_weights = self.predicate_cross_attn(
                reason_nodes,
                predicate_tokens,
                predicate_tokens,
                need_weights=True,
                average_attn_weights=False,
            )
            gate = torch.sigmoid(self.predicate_gate).clamp(max=0.20).view(1, self.reason_dim, 1)
            label_nodes = torch.cat([label_nodes[:, : self.action_dim], reason_nodes + gate * pred_delta], dim=1)
            reason_predicate_attention = pred_weights.detach()
        else:
            reason_predicate_attention = torch.zeros(b, 4, self.reason_dim, 1, device=patch.device, dtype=patch.dtype)

        legacy_label_nodes = label_nodes
        legacy_reason_logits, legacy_action_visual, legacy_action_reason, legacy_gate, legacy_direct = self._action_outputs(legacy_label_nodes)
        action_nodes_legacy = legacy_label_nodes[:, : self.action_dim]
        reason_nodes = legacy_label_nodes[:, self.action_dim:]
        seca_out = self.seca(action_nodes_legacy, reason_nodes)
        if self.seca_enabled:
            action_nodes_final = seca_out["action_nodes_seca"]
        else:
            action_nodes_final = action_nodes_legacy
            # Preserve tensor shapes while making the disabled path exactly legacy.
            seca_out["action_nodes_seca"] = action_nodes_legacy
            seca_out["residual_scale"] = torch.zeros_like(seca_out["residual_scale"])
        label_nodes = torch.cat([action_nodes_final, reason_nodes], dim=1)
        reason_logits_visual, action_visual_logits, action_reason_logits, gate, action_logits_direct = self._action_outputs(label_nodes)
        return {
            "label_nodes": label_nodes,
            "label_attention": attn,
            "action_nodes_legacy": action_nodes_legacy,
            "action_nodes_seca": seca_out["action_nodes_seca"],
            "action_visual_logits_legacy": legacy_action_visual,
            "action_reason_logits_legacy": legacy_action_reason,
            "action_fusion_gate_legacy": legacy_gate,
            "action_logits_direct_legacy": legacy_direct,
            "reason_logits_visual_legacy": legacy_reason_logits,
            "action_visual_logits": action_visual_logits,
            "action_reason_logits": action_reason_logits,
            "reason_logits_visual": reason_logits_visual,
            "action_fusion_gate": gate,
            "action_token_norm_mean": action_nodes_final.norm(dim=-1).mean(),
            "predicate_conditioning_strength": torch.sigmoid(self.predicate_gate).detach(),
            "reason_predicate_attention": reason_predicate_attention,
            "action_logits_direct": action_logits_direct,
            "seca_enabled": torch.tensor(float(self.seca_enabled), device=patch.device),
            "seca_action_reason_attention_heads": seca_out["action_reason_attention_heads"],
            "seca_action_reason_attention": seca_out["action_reason_attention"],
            "seca_action_reason_attention_no_null": seca_out["action_reason_attention_no_null"],
            "seca_null_attention": seca_out["null_attention"],
            "seca_residual_scale": seca_out["residual_scale"],
            "seca_evidence_context": seca_out["evidence_context"],
            "seca_evidence_context_norm": seca_out["evidence_context_norm"],
            "seca_active_reason_count": seca_out["active_reason_count"],
            "seca_attention_entropy": seca_out["attention_entropy"],
            "seca_action_attention_diversity": seca_out["action_attention_diversity"],
        }
