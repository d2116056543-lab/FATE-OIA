from __future__ import annotations

import torch
from torch import nn

from .eagle_pu_sparse_ops import entmax15_bisect


class ReasonPrototypeTransport(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21, num_prototypes: int = 6) -> None:
        super().__init__()
        self.reason_dim = reason_dim
        self.num_prototypes = num_prototypes
        self.reason_prototypes = nn.Parameter(torch.randn(reason_dim, num_prototypes, dim) * 0.02)
        self.state_proj = nn.Linear(dim, dim)
        self.delta_head = nn.Linear(dim, 1)
        self.gate_head = nn.Linear(dim, 1)

    def forward(self, reason_nodes: torch.Tensor, state_tokens: torch.Tensor, epoch: int = 0) -> dict[str, torch.Tensor]:
        b, r, d = reason_nodes.shape
        states = self.state_proj(state_tokens).mean(1)
        proto = self.reason_prototypes.unsqueeze(0).expand(b, -1, -1, -1)
        query = reason_nodes.unsqueeze(2) + states.view(b, 1, 1, d)
        scores = (query * proto).sum(-1) / (d ** 0.5)
        weights = torch.softmax(scores, dim=-1) if epoch < 4 else entmax15_bisect(scores, dim=-1)
        transported = (weights.unsqueeze(-1) * proto).sum(2)
        raw_delta = self.delta_head(transported).squeeze(-1)
        delta = 0.12 * torch.tanh(raw_delta / 0.12)
        gate = torch.sigmoid(self.gate_head(reason_nodes)).squeeze(-1)
        return {"prototype_reason_delta": delta, "proto_gate": gate, "prototype_weights": weights}
