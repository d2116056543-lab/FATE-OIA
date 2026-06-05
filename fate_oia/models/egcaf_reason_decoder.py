from __future__ import annotations

import torch
from torch import nn


class ReasonFromFactorDecoder(nn.Module):
    def __init__(self, hidden_dim: int = 256, reason_dim: int = 21, action_dim: int = 4, tail_indices: tuple[int, ...] = (12, 9, 5, 14, 6, 11, 10, 13)) -> None:
        super().__init__()
        self.reason_queries = nn.Parameter(torch.randn(reason_dim, hidden_dim) * 0.02)
        self.cross = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.self_attn = nn.TransformerEncoderLayer(hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2, batch_first=True, dropout=0.0)
        self.head = nn.Linear(hidden_dim, 1)
        self.action_dim = int(action_dim)
        self.tail_indices = tuple(int(x) for x in tail_indices)
        self.tail_adapter = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.tail_gate = nn.Parameter(torch.tensor(0.2))

    def forward(self, selected_embeddings: torch.Tensor, scene_state_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        b, action_dim, k, d = selected_embeddings.shape
        selected_memory = selected_embeddings.reshape(b, action_dim * k, d)
        memory = torch.cat([selected_memory, scene_state_tokens], 1)
        query = self.reason_queries.unsqueeze(0).expand(b, -1, -1)
        x, attn = self.cross(query, memory, memory, need_weights=True)
        x = self.self_attn(x)
        logits = self.head(x).squeeze(-1)
        if self.tail_indices:
            delta = self.tail_adapter(x[:, list(self.tail_indices)]).squeeze(-1) * torch.sigmoid(self.tail_gate)
            logits = logits.clone()
            logits[:, list(self.tail_indices)] += delta
        reason_factor_attention = attn[:, :, : action_dim * k].reshape(b, -1, action_dim, k)
        flat_top = reason_factor_attention.reshape(b, reason_factor_attention.shape[1], -1).argmax(-1)
        reason_support_per_selected_factor = reason_factor_attention.sum(1)
        return {
            "reason_logits": logits,
            "reason_attention": attn,
            "reason_memory_tokens": memory,
            "reason_factor_attention": reason_factor_attention,
            "top_factor_for_each_reason": flat_top,
            "reason_support_per_selected_factor": reason_support_per_selected_factor,
        }

