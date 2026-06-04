from __future__ import annotations

import torch
from torch import nn


class ExpertAdapterBlock(nn.Module):
    def __init__(self, dim: int = 384, num_heads: int = 4, ffn_ratio: float = 2.0, dropout: float = 0.05) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        hidden = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, label_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        x = label_tokens
        y, _ = self.self_attn(self.self_norm(x), self.self_norm(x), self.self_norm(x), need_weights=False)
        x = x + y
        z, _ = self.cross_attn(self.cross_norm(x), self.context_norm(context_tokens), self.context_norm(context_tokens), need_weights=False)
        x = x + z
        x = x + self.ffn(x)
        return x


class _BaseExpert(nn.Module):
    def __init__(self, dim: int, label_dim: int, depth: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ExpertAdapterBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))
        self.label_dim = label_dim

    def forward_tokens(self, label_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        x = label_tokens
        for block in self.blocks:
            x = block(x, context_tokens)
        return x

    def forward(self, label_tokens: torch.Tensor, context_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.forward_tokens(label_tokens, context_tokens)
        return {"tokens": x, "logits": self.head(x).squeeze(-1)}


class ActionExpert(_BaseExpert):
    def __init__(self, dim: int = 384, action_dim: int = 4, depth: int = 2, heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__(dim, action_dim, depth, heads, dropout)


class ReasonExpert(_BaseExpert):
    def __init__(self, dim: int = 384, reason_dim: int = 21, depth: int = 2, heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__(dim, reason_dim, depth, heads, dropout)


class SharedExpert(nn.Module):
    def __init__(self, dim: int = 384, depth: int = 1, heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([ExpertAdapterBlock(dim, heads, dropout=dropout) for _ in range(depth)])

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = tokens
        for block in self.blocks:
            x = block(x, context)
        return x


class TailExpert(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21, tail_indices: list[int] | None = None, depth: int = 1, heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        self.reason_dim = reason_dim
        self.tail_indices = list(tail_indices or [5, 6, 9, 10, 11, 12, 13, 14])
        self.blocks = nn.ModuleList([ExpertAdapterBlock(dim, heads, dropout=dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, reason_tokens: torch.Tensor, context_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if not self.tail_indices:
            return {"tail_delta": reason_tokens.new_zeros(reason_tokens.shape[0], self.reason_dim)}
        tail = reason_tokens[:, self.tail_indices]
        for block in self.blocks:
            tail = block(tail, context_tokens)
        tail_logits = self.head(tail).squeeze(-1)
        delta = reason_tokens.new_zeros(reason_tokens.shape[0], self.reason_dim)
        delta[:, self.tail_indices] = tail_logits
        return {"tail_delta": delta, "tail_tokens": tail}
