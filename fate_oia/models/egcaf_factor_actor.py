from __future__ import annotations

import torch
from torch import nn


class FactorActor(nn.Module):
    def __init__(self, hidden_dim: int = 256, action_dim: int = 4, residual_cap: float = 0.03, residual_enabled_default: bool = False) -> None:
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
        self.residual_enabled_default = bool(residual_enabled_default)

    def summarize(self, selected_embeddings: torch.Tensor, selected_weights: torch.Tensor) -> torch.Tensor:
        b, a, k, d = selected_embeddings.shape
        weights = selected_weights / (selected_weights.sum(-1, keepdim=True) + 1e-6)
        weighted = (selected_embeddings * weights.unsqueeze(-1)).sum(2)
        return self.action_interact(weighted + self.action_queries.view(1, a, d))

    def _logits_from_summaries(self, summaries: torch.Tensor, residual_enabled: bool, residual_gate: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        core = self.core_head(summaries).squeeze(-1)
        proto = self.prototype_vectors + torch.tanh(self.prototype_residual) * 0.1
        proto_context = torch.softmax(self.proto_gate(summaries.mean(1)), -1) @ proto
        core = core + 0.10 * proto_context
        delta = torch.tanh(self.delta_head(summaries).squeeze(-1)) * self.residual_cap
        if residual_gate is None:
            residual_gate = torch.ones_like(delta) if residual_enabled else torch.zeros_like(delta)
        if not residual_enabled:
            residual_gate = torch.zeros_like(delta)
        final = core + delta * residual_gate
        return {
            "action_core_logits": core,
            "action_final_logits": final,
            "residual_delta": delta,
            "residual_gate": residual_gate,
        }

    def forward(self, selected_embeddings: torch.Tensor, selected_weights: torch.Tensor, residual_enabled: bool | None = None, residual_gate: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        summaries = self.summarize(selected_embeddings, selected_weights)
        enabled = self.residual_enabled_default if residual_enabled is None else bool(residual_enabled)
        return self._logits_from_summaries(summaries, enabled, residual_gate)

    def mode_logits(
        self,
        factor_embeddings: torch.Tensor,
        factor_weights: torch.Tensor,
        selected_indices: torch.Tensor,
        mode: str,
        random_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Action logits under all/selected/without-selected/without-random factor modes."""
        b, m, d = factor_embeddings.shape
        a = factor_weights.shape[1]
        weights = factor_weights.clone()
        if mode == "selected":
            mask = torch.zeros_like(weights)
            mask.scatter_(-1, selected_indices, 1.0)
            weights = weights * mask
        elif mode == "without-selected":
            mask = torch.ones_like(weights)
            mask.scatter_(-1, selected_indices, 0.0)
            weights = weights * mask
        elif mode == "without-random":
            if random_indices is None:
                raise ValueError("random_indices are required for without-random mode")
            mask = torch.ones_like(weights)
            mask.scatter_(-1, random_indices, 0.0)
            weights = weights * mask
        elif mode == "all":
            pass
        else:
            raise ValueError(f"Unsupported FactorActor mode: {mode}")
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-6)
        summaries = torch.einsum("bam,bmd->bad", weights, factor_embeddings)
        summaries = self.action_interact(summaries + self.action_queries.view(1, a, d))
        return self._logits_from_summaries(summaries, residual_enabled=False)["action_core_logits"]

