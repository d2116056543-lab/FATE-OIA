from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .cage_label_nodes import CAGELabelNodes
from .cage_evidence_retriever import CAGEEvidenceRetriever
from .cage_dynamic_transport import CAGEDynamicTransport
from .cage_reason_reliability import CAGEReasonReliability


class CAGEOIAFeatureModel(nn.Module):
    """Category-grounded evidence transport model over pre-extracted tokens."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        action_dim: int = 4,
        reason_dim: int = 21,
        evidence_topk: int = 8,
        transport_steps: int = 1,
        residual_cap: float = 2.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.num_labels = action_dim + reason_dim
        self.residual_cap = float(residual_cap)
        self.token_proj = nn.Linear(input_dim, hidden_dim)
        self.label_nodes = CAGELabelNodes(action_dim=action_dim, reason_dim=reason_dim, hidden_dim=hidden_dim)
        self.base_norm = nn.LayerNorm(hidden_dim)
        self.base_action_head = nn.Linear(hidden_dim, action_dim)
        self.base_reason_head = nn.Linear(hidden_dim, reason_dim)
        self.evidence = CAGEEvidenceRetriever(hidden_dim, self.num_labels, topk=evidence_topk)
        self.transport = CAGEDynamicTransport(hidden_dim, action_dim, reason_dim, num_steps=transport_steps)
        self.action_node_head = nn.Linear(hidden_dim, 1)
        self.reason_node_head = nn.Linear(hidden_dim, 1)
        self.action_gate_head = nn.Linear(hidden_dim, action_dim)
        nn.init.constant_(self.action_gate_head.bias, -2.0)
        self.reason_reliability = CAGEReasonReliability(reason_dim, hidden_dim=max(16, hidden_dim // 4))
        self.register_buffer("reason_label_frequency", torch.ones(reason_dim) * 0.5)

    def forward(self, tokens: torch.Tensor, cooccur_prior: torch.Tensor | None = None) -> Dict[str, torch.Tensor | Dict]:
        if tokens.dim() != 3:
            raise ValueError("tokens must be [B,N,D]")
        hidden = self.token_proj(tokens)
        pooled = self.base_norm(hidden.mean(dim=1))
        base_action = self.base_action_head(pooled)
        base_reason = self.base_reason_head(pooled)
        base_all = torch.cat([base_action, base_reason], dim=-1)

        label_node_out = self.label_nodes(batch_size=tokens.shape[0])
        evidence = self.evidence(hidden, label_node_out["label_queries"])
        transport = self.transport(evidence["evidence_state"], base_logits=base_all, cooccur_prior=cooccur_prior)
        label_state = transport["updated_label_state"]
        action_state = label_state[:, : self.action_dim]
        reason_state = label_state[:, self.action_dim :]
        transport_action = self.action_node_head(action_state).squeeze(-1)
        transport_reason = self.reason_node_head(reason_state).squeeze(-1)

        action_gate = torch.sigmoid(self.action_gate_head(pooled))
        action_delta = (transport_action - base_action).clamp(-self.residual_cap, self.residual_cap)
        action_logits = base_action + action_gate * action_delta

        reason_conf = evidence["evidence_confidence"][:, self.action_dim :]
        selected_drop = torch.zeros_like(reason_conf)
        reliability = self.reason_reliability(base_reason, reason_conf, selected_drop, self.reason_label_frequency)["reason_reliability"]
        reason_delta = (transport_reason - base_reason).clamp(-self.residual_cap, self.residual_cap)
        reason_logits = base_reason + reliability * reason_delta

        return {
            "action_logits": action_logits,
            "reason_logits": reason_logits,
            "base_action_logits": base_action,
            "base_reason_logits": base_reason,
            "transport_action_logits": transport_action,
            "transport_reason_logits": transport_reason,
            "action_gate": action_gate,
            "reason_reliability": reliability,
            "selected_vs_random_ready": torch.tensor(False, device=tokens.device),
            "label_nodes": label_node_out,
            "evidence": evidence,
            "transport": transport,
        }
