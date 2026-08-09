from __future__ import annotations

import torch
from torch import Tensor, nn


class PACTContextDecoder(nn.Module):
    """Action-owned copy of the source trunk's semantic interaction block."""

    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21) -> None:
        super().__init__()
        self.action_dim, self.reason_dim = int(action_dim), int(reason_dim)
        self.label_self_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.predicate_cross_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.predicate_gate = nn.Parameter(torch.full((reason_dim,), -2.944))
        self.logit_head = nn.Linear(dim, 1)
        self.reason_to_action = nn.Linear(reason_dim, action_dim)
        hidden = max(dim // 2, 1)
        self.action_visual_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.fusion_gate = nn.Linear(dim * 2, action_dim)

    def forward(self, shared_nodes: Tensor, predicate_tokens: Tensor | None) -> dict[str, Tensor]:
        nodes = shared_nodes + self.label_self_attn(shared_nodes, shared_nodes, shared_nodes, need_weights=False)[0]
        if predicate_tokens is not None:
            reason = nodes[:, self.action_dim:]
            delta = self.predicate_cross_attn(reason, predicate_tokens, predicate_tokens, need_weights=False)[0]
            gate = torch.sigmoid(self.predicate_gate).clamp(max=0.20).view(1, self.reason_dim, 1)
            nodes = torch.cat((nodes[:, :self.action_dim], reason + gate * delta), 1)
        logits = self.logit_head(nodes).squeeze(-1)
        reason_context = logits[:, self.action_dim:]
        action_nodes = nodes[:, :self.action_dim]
        visual = self.action_visual_head(action_nodes).squeeze(-1)
        context = self.reason_to_action(reason_context)
        gate_input = torch.cat((action_nodes.mean(1), nodes[:, self.action_dim:].mean(1)), -1)
        fusion = torch.sigmoid(self.fusion_gate(gate_input)).clamp(0.10, 0.90)
        action = fusion * visual + (1.0 - fusion) * context
        return {"context_label_nodes": nodes, "action_nodes_context": action_nodes,
                "reason_context_logits": reason_context, "action_visual_logits": visual,
                "action_context_logits": context, "action_fusion_gate": fusion,
                "action_logits_primary": action}
