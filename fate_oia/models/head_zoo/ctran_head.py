from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.head_zoo.base import BaseOIAHead, with_common_fields


class CTranMaskedHead(BaseOIAHead):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        reveal_prob: float = 0.35,
        num_heads: int = 6,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.num_labels = self.action_dim + self.reason_dim
        self.action_reveal_prob = float(reveal_prob)
        self.reason_reveal_prob = float(reveal_prob)
        self.reveal_prob = float(reveal_prob)
        self.label_embed = nn.Embedding(self.num_labels, dim)
        self.state_embed = nn.Embedding(3, dim)  # negative, positive, unknown
        self.visual_proj = nn.Linear(dim, dim)
        layer = nn.TransformerEncoderLayer(dim, num_heads, dim * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.cls = nn.Linear(dim, 1)

    def _states(self, labels: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
        if (not self.training) or labels is None:
            return torch.full((batch, self.num_labels), 2, dtype=torch.long, device=device)
        action_reveal = torch.rand(batch, self.action_dim, device=device) < self.action_reveal_prob
        reason_reveal = torch.rand(batch, self.reason_dim, device=device) < self.reason_reveal_prob
        reveal = torch.cat([action_reveal, reason_reveal], dim=1)
        self.last_reveal_mask = reveal.detach()
        values = torch.where(labels.float() > 0, torch.ones_like(labels, dtype=torch.long), torch.zeros_like(labels, dtype=torch.long))
        return torch.where(reveal, values, torch.full_like(values, 2))

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor | None = None, **kwargs):
        b = tokens.shape[0]
        device = tokens.device
        summary = self.visual_proj(tokens.mean(dim=1, keepdim=True))
        label_ids = torch.arange(self.num_labels, device=device).unsqueeze(0).expand(b, -1)
        states = self._states(labels, b, device)
        label_tokens = self.label_embed(label_ids) + self.state_embed(states)
        encoded = self.encoder(torch.cat([summary, label_tokens], dim=1))
        label_tokens = encoded[:, 1:]
        logits = self.cls(label_tokens).squeeze(-1)
        attention = torch.softmax(torch.einsum("bld,bnd->bln", label_tokens, tokens) / (tokens.shape[-1] ** 0.5), dim=-1)
        return with_common_fields(logits, self.action_dim, label_tokens=label_tokens, attention=attention, aux_losses={})
