from __future__ import annotations

import torch
from torch import nn

from .cluster_semantics import load_exp29_names
from .types import Exp29Output


class Exp29Head(nn.Module):
    def __init__(self, dim: int = 384, exp_dim: int = 29, label_names_path: str | None = None) -> None:
        super().__init__()
        self.label_names = load_exp29_names(label_names_path)
        self.queries = nn.Parameter(torch.randn(exp_dim, dim) * 0.02)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.logit = nn.Linear(dim, 1)

    def forward(self, factor_tokens: torch.Tensor, predicate_tokens: torch.Tensor) -> Exp29Output:
        source = torch.cat([factor_tokens, predicate_tokens], dim=1)
        d = source.shape[-1]
        q = self.queries.view(1, -1, d)
        score = torch.einsum("bld,bnd->bln", q, self.key(source)) / (d ** 0.5)
        attn = torch.softmax(score, dim=-1)
        label_tokens = torch.einsum("bln,bnd->bld", attn, self.value(source))
        logits = self.logit(label_tokens).squeeze(-1)
        probs = torch.sigmoid(logits)
        stats = {
            "exp29_attention_entropy": float((-(attn.clamp_min(1e-9).log() * attn).sum(-1)).mean().detach().cpu()),
            "exp29_positive_rate": float((probs > 0.5).float().mean().detach().cpu()),
        }
        return Exp29Output(
            logits=logits,
            probs=probs,
            label_mask=torch.ones_like(logits),
            label_names=self.label_names,
            attention=attn,
            stats=stats,
        )

