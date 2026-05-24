from __future__ import annotations

import torch
from torch import nn


class LabelQueryHead(nn.Module):
    def __init__(self, dim: int, num_labels: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_labels, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.cls = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        b = tokens.shape[0]
        q = self.queries.unsqueeze(0).expand(b, -1, -1)
        out, attn = self.attn(q, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
        out = self.norm(out)
        logits = self.cls(out).squeeze(-1)
        return {"logits": logits, "label_tokens": out, "attention": attn}
