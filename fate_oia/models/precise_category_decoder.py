from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PRECISECategoryDecoder(nn.Module):
    def __init__(self, reason_schema: list[dict[str, Any]], action_schema: list[dict[str, Any]], dim: int = 384, heads: int = 4) -> None:
        super().__init__()
        self.reason_schema = reason_schema
        self.action_schema = action_schema
        self.entity = nn.Embedding(len({row["entity"] for row in reason_schema}), dim)
        self.state = nn.Embedding(len({row["state"] for row in reason_schema}), dim)
        self.sector = nn.Embedding(len({row["sector"] for row in reason_schema}), dim)
        self.role = nn.Embedding(len({row["decision_role"] for row in reason_schema}), dim)
        self.reason_residual = nn.Embedding(21, dim)
        self.entity_ids = self._ids("entity")
        self.state_ids = self._ids("state")
        self.sector_ids = self._ids("sector")
        self.role_ids = self._ids("decision_role")
        self.forward_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.stop_query = nn.Parameter(torch.randn(dim) * 0.02)
        self.side_shared = nn.Parameter(torch.randn(dim) * 0.02)
        self.left_embedding = nn.Parameter(torch.randn(dim) * 0.02)
        self.right_embedding = nn.Parameter(torch.randn(dim) * 0.02)
        self.action_cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.reason_cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.action_self = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.reason_self = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.action_head = nn.Linear(dim, 1)
        self.reason_head = nn.Linear(dim, 1)

    def _ids(self, key: str) -> torch.Tensor:
        values = {value: idx for idx, value in enumerate(sorted({row[key] for row in self.reason_schema}))}
        return torch.tensor([values[row[key]] for row in self.reason_schema], dtype=torch.long)

    def action_queries(self) -> torch.Tensor:
        base = {"forward": self.forward_query, "stop": self.stop_query, "side_shared": self.side_shared}
        side = {"none": 0.0, "left": self.left_embedding, "right": self.right_embedding}
        return torch.stack([base[row["query_base"]] + side[row["side_embedding"]] for row in self.action_schema])

    def reason_queries(self) -> torch.Tensor:
        device = self.reason_residual.weight.device
        ids = torch.arange(21, device=device)
        return self.entity(self.entity_ids.to(device)) + self.state(self.state_ids.to(device)) + self.sector(self.sector_ids.to(device)) + self.role(self.role_ids.to(device)) + self.reason_residual(ids)

    @staticmethod
    def _entropy(weights: torch.Tensor) -> torch.Tensor:
        return -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)

    def first_pass(self, action_queries: torch.Tensor, reason_queries: torch.Tensor, action_context_tokens: torch.Tensor, reason_context_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = action_context_tokens.shape[0]
        action_seed = action_queries.unsqueeze(0).expand(batch, -1, -1)
        reason_seed = reason_queries.unsqueeze(0).expand(batch, -1, -1)
        action_cross, action_weights = self.action_cross(action_seed, action_context_tokens, action_context_tokens, need_weights=True, average_attn_weights=False)
        reason_cross, reason_weights = self.reason_cross(reason_seed, reason_context_tokens, reason_context_tokens, need_weights=True, average_attn_weights=False)
        action_tokens, _ = self.action_self(action_cross, action_cross, action_cross, need_weights=False)
        reason_tokens, _ = self.reason_self(reason_cross, reason_cross, reason_cross, need_weights=False)
        action_entropy = self._entropy(action_weights.mean(1))
        reason_entropy = self._entropy(reason_weights.mean(1))
        return {
            "action_tokens_direct": action_tokens,
            "reason_tokens_direct": reason_tokens,
            "action_logits_direct": self.action_head(action_tokens).squeeze(-1),
            "reason_logits_direct": self.reason_head(reason_tokens).squeeze(-1),
            "action_entropy": action_entropy,
            "reason_entropy": reason_entropy,
        }
