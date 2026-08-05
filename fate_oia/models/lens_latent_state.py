from __future__ import annotations

import torch
from torch import Tensor, nn


class LENSLatentState(nn.Module):
    def __init__(self, dim: int = 384, reason_dim: int = 21) -> None:
        super().__init__()
        self.reason_dim = reason_dim
        self.support_delta = nn.Sequential(nn.Linear(dim * 2 + 3, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.support_delta[-1].weight)
        nn.init.zeros_(self.support_delta[-1].bias)
        self.unknown_mlp = nn.Sequential(nn.Linear(dim * 2 + 3, dim), nn.GELU(), nn.Linear(dim, 1))
        self.state_embedding = nn.Parameter(torch.randn(reason_dim, 3, dim) * 0.02)

    def forward(self, reason_nodes: Tensor, evidence_token: Tensor, source_reason_visual: Tensor, null_mass: Tensor, entropy: Tensor, snr: Tensor, *, progress: float) -> dict[str, Tensor]:
        features = torch.cat([reason_nodes, evidence_token, null_mass.unsqueeze(-1), entropy.unsqueeze(-1), snr.unsqueeze(-1)], dim=-1)
        alpha = float(max(0.0, min(1.0, progress)))
        delta = self.support_delta(features).squeeze(-1)
        support = source_reason_visual + alpha * delta
        unknown = alpha * torch.sigmoid(self.unknown_mlp(features).squeeze(-1))
        positive = (1.0 - unknown) * torch.sigmoid(support)
        counter = (1.0 - unknown) * (1.0 - torch.sigmoid(support))
        state_prob = torch.stack([positive, counter, unknown], dim=-1)
        state_token = evidence_token + torch.einsum("brs,rsd->brd", state_prob, self.state_embedding)
        return {
            "state_prob": state_prob,
            "state_positive_prob": positive,
            "state_counter_prob": counter,
            "state_unknown_prob": unknown,
            "state_observability": 1.0 - unknown,
            "state_support_logit": support,
            "state_token": state_token,
            "state_embeddings": self.state_embedding,
        }
