from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class CAGEDynamicTransport(nn.Module):
    """Typed sample-specific action/reason evidence transport graph."""

    def __init__(self, hidden_dim: int, action_dim: int = 4, reason_dim: int = 21, num_steps: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.num_labels = action_dim + reason_dim
        self.num_steps = max(int(num_steps), 1)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.msg_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.logit_bias = nn.Linear(2, 1)

    def _edge_logits(self, state: torch.Tensor, base_logits: torch.Tensor | None, cooccur_prior: torch.Tensor | None) -> torch.Tensor:
        q = self.q_proj(state)
        k = self.k_proj(state)
        edge = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(float(self.hidden_dim))
        if base_logits is not None:
            pair = torch.stack(
                [base_logits.unsqueeze(2).expand(-1, -1, self.num_labels), base_logits.unsqueeze(1).expand(-1, self.num_labels, -1)],
                dim=-1,
            )
            edge = edge + self.logit_bias(pair).squeeze(-1)
        if cooccur_prior is not None:
            edge = edge + cooccur_prior.to(edge.device, edge.dtype).unsqueeze(0)
        return edge

    def forward(
        self,
        label_state: torch.Tensor,
        *,
        base_logits: torch.Tensor | None = None,
        cooccur_prior: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor | Dict[str, torch.Tensor]]:
        if label_state.dim() != 3:
            raise ValueError("label_state must be [B,L,D]")
        if label_state.shape[1] != self.num_labels:
            raise ValueError(f"expected {self.num_labels} labels, got {label_state.shape[1]}")
        state = label_state
        edge = None
        attn = None
        for _ in range(self.num_steps):
            edge = self._edge_logits(state, base_logits, cooccur_prior)
            attn = F.softmax(edge, dim=-1)
            msg = torch.matmul(attn, self.v_proj(state))
            state = self.norm(state + self.msg_proj(msg))
        assert edge is not None and attn is not None
        a = self.action_dim
        typed = {
            "A_A": attn[:, :a, :a],
            "A_R": attn[:, :a, a:],
            "R_A": attn[:, a:, :a],
            "R_R": attn[:, a:, a:],
        }
        return {"updated_label_state": state, "edge_matrix": attn, "edge_logits": edge, "typed_edges": typed}
