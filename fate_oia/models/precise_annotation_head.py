from __future__ import annotations

import torch
from torch import nn


class PRECISEAnnotationHead(nn.Module):
    """Sample-dependent observed-label adapter isolated from semantic vision paths."""

    def __init__(self, dim: int = 384, rank: int = 8, delta_cap: float = 0.75) -> None:
        super().__init__()
        self.delta_cap = delta_cap
        self.down = nn.Sequential(nn.Linear(dim * 2, rank), nn.GELU())
        self.up = nn.Linear(rank, 1)
        self.bias = nn.Parameter(torch.zeros(21))

    def forward(self, reason_tokens_semantic: torch.Tensor, global_context: torch.Tensor, semantic_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        context = global_context.detach().unsqueeze(1).expand(-1, reason_tokens_semantic.shape[1], -1)
        hidden = self.down(torch.cat([reason_tokens_semantic.detach(), context], dim=-1))
        delta = self.delta_cap * torch.tanh(self.up(hidden).squeeze(-1) + self.bias.view(1, -1))
        observed = semantic_logits.detach() + delta
        return {"reason_logits_observed": observed, "annotation_delta": delta}
