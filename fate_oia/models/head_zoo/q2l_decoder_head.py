from __future__ import annotations

import torch

from fate_oia.models.head_zoo.base import BaseOIAHead
from fate_oia.models.semantic_label_queries import SemanticLabelQueries
from fate_oia.models.strong_label_decoder import StrongLabelDecoder


class Q2LDecoderHead(BaseOIAHead):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        num_heads: int = 6,
        self_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.queries = SemanticLabelQueries(action_dim + reason_dim, dim, action_dim=action_dim, dropout=dropout)
        self.decoder = StrongLabelDecoder(dim=dim, action_dim=action_dim, reason_dim=reason_dim, num_heads=num_heads, self_layers=self_layers, dropout=dropout)

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        q = self.queries(tokens.shape[0], device=tokens.device)
        out = self.decoder(q, tokens)
        out["aux_losses"] = {}
        return out
