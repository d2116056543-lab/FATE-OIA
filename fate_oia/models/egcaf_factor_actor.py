from __future__ import annotations

import torch
from torch import nn


class FactorActor(nn.Module):
    def __init__(self, hidden_dim: int = 256, action_dim: int = 4, residual_cap: float = 0.03) -> None:
        super().__init__()
        self.action_queries = nn.Parameter(torch.randn(action_dim, hidden_dim) * 0.02)
        self.action_interact = nn.TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True, dropout=0.0)
        self.core_head = nn.Linear(hidden_dim, 1)
        proto = torch.tensor([[0,1,0,0],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[0,1,0,1],[0,1,1,0],[1,1,0,0]], dtype=torch.float32)
        self.register_buffer("prototype_vectors", proto)
        self.prototype_residual = nn.Parameter(torch.zeros_like(proto))
        self.proto_gate = nn.Linear(hidden_dim, proto.shape[0])
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.residual_cap = float(residual_cap)

    def summarize(self, selected_embeddings: torch.Tensor, selected_weights: torch.Tensor) -> torch.Tensor:
        b, a, k, d = selected_embeddings.shape
        weights = selected_weights / (selected_weights.sum(-1, keepdim=True) + 1e-6)
        weighted = (selected_embeddings * weights.unsqueeze(-1)).sum(2)
        return self.action_interact(weighted + self.action_queries.view(1, a, d))

    def forward(self, selected_embeddings: torch.Tensor, selected_weights: torch.Tensor, residual_enabled: bool = True) -> dict[str, torch.Tensor]:
        summaries = self.summarize(selected_embeddings, selected_weights)
        core = self.core_head(summaries).squeeze(-1)
        proto = self.prototype_vectors + torch.tanh(self.prototype_residual) * 0.1
        proto_context = torch.softmax(self.proto_gate(summaries.mean(1)), -1) @ proto
        core = core + 0.10 * proto_context
        delta = torch.tanh(self.delta_head(summaries).squeeze(-1)) * self.residual_cap
        final = core + (delta if residual_enabled else 0.0)
        return {
            "action_core_logits": core,
            "action_final_logits": final,
            "residual_delta": delta,
            "residual_gate": torch.ones_like(delta) if residual_enabled else torch.zeros_like(delta),
        }
