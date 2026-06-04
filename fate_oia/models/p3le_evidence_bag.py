from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.p3le_sparse_attention import SparseRegionAttention


class WeakEvidenceBagRegularizer(nn.Module):
    """Weak evidence diagnostic/regularizer.

    It never modifies action logits. If selected evidence is not stronger than
    the random baseline, `evidence_lambda_active` is zero and training receives
    diagnostics only.
    """

    def __init__(self, dim: int = 384, reason_dim: int = 21, topk: int = 64) -> None:
        super().__init__()
        self.reason_dim = int(reason_dim)
        self.sparse_attention = SparseRegionAttention(dim, topk=topk)
        self.score = nn.Linear(dim, 1)

    def forward(self, tokens: torch.Tensor, reason_tokens: torch.Tensor, reason_labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        sparse = self.sparse_attention(reason_tokens, tokens)
        selected_score = self.score(sparse["pooled"]).squeeze(-1)
        # Deterministic random-like baseline: roll token order by half length.
        rolled = tokens.roll(shifts=max(1, tokens.shape[1] // 2), dims=1)
        random_sparse = self.sparse_attention(reason_tokens.detach(), rolled)
        random_score = self.score(random_sparse["pooled"]).squeeze(-1).detach()
        selected_mean = selected_score.sigmoid().mean()
        random_mean = random_score.sigmoid().mean()
        active = (selected_mean > random_mean).to(tokens.dtype)
        if reason_labels is None:
            loss = selected_score.new_zeros(())
        else:
            # Weak ranking loss only. It is turned off by active=0.
            margin = 0.05
            loss = torch.relu(random_score - selected_score + margin)
            loss = (loss * reason_labels.float()).mean() * active
        return {
            "evidence_selected_score": selected_score,
            "evidence_random_score": random_score,
            "evidence_selected_mean": selected_mean,
            "evidence_random_mean": random_mean,
            "evidence_lambda_active": active,
            "evidence_loss": loss,
            "evidence_indices": sparse["indices"],
        }
