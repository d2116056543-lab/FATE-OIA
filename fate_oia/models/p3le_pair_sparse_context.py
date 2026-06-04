from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.p3le_sparse_attention import SparseRegionAttention


class PairSparseContext(nn.Module):
    """Sparse visual evidence context for pair-aware action-reason scoring."""

    def __init__(self, dim: int = 384, topk: int = 64) -> None:
        super().__init__()
        self.action_sparse = SparseRegionAttention(dim, topk=topk)
        self.reason_sparse = SparseRegionAttention(dim, topk=topk)

    def forward(self, action_tokens: torch.Tensor, reason_tokens: torch.Tensor, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        action = self.action_sparse(action_tokens, tokens)
        reason = self.reason_sparse(reason_tokens, tokens)
        return {
            "action_sparse_context": action["pooled"],
            "reason_sparse_context": reason["pooled"],
            "action_sparse_indices": action["indices"],
            "reason_sparse_indices": reason["indices"],
            "action_sparse_weight_mean": action["weights"].mean(),
            "reason_sparse_weight_mean": reason["weights"].mean(),
            "action_sparse_topk": action["indices"].new_tensor(action["indices"].shape[-1]).float(),
            "reason_sparse_topk": reason["indices"].new_tensor(reason["indices"].shape[-1]).float(),
        }
