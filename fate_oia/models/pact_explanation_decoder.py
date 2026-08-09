from __future__ import annotations

import torch
from torch import Tensor, nn


class PACTExplanationDecoder(nn.Module):
    """Explanation-owned source-equivalent interaction over all 25 label nodes."""

    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21) -> None:
        super().__init__()
        self.action_dim, self.reason_dim = int(action_dim), int(reason_dim)
        self.label_self_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.predicate_cross_attn = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.predicate_gate = nn.Parameter(torch.full((reason_dim,), -2.944))
        self.logit_head = nn.Linear(dim, 1)

    def forward(self, shared_nodes: Tensor, predicate_tokens: Tensor | None) -> dict[str, Tensor]:
        nodes = shared_nodes + self.label_self_attn(shared_nodes, shared_nodes, shared_nodes, need_weights=False)[0]
        if predicate_tokens is not None:
            reason = nodes[:, self.action_dim:]
            delta = self.predicate_cross_attn(reason, predicate_tokens, predicate_tokens, need_weights=False)[0]
            gate = torch.sigmoid(self.predicate_gate).clamp(max=0.20).view(1, self.reason_dim, 1)
            nodes = torch.cat((nodes[:, :self.action_dim], reason + gate * delta), 1)
        logits = self.logit_head(nodes).squeeze(-1)
        return {"explanation_label_nodes": nodes, "reason_nodes_formal": nodes[:, self.action_dim:],
                "reason_logits_visual_formal": logits[:, self.action_dim:]}
