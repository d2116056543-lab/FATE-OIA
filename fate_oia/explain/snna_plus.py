from __future__ import annotations

import torch

from fate_oia.models.token_provenance import recover_attribution


def label_attention_to_token_scores(attention: torch.Tensor, label_index: int, provenance: torch.Tensor | None = None) -> torch.Tensor:
    """Extract per-token attribution for one label from label-query attention.

    Supports attention [B,H,L,N] or [B,L,N]. If provenance is provided, reduced
    attribution is recovered to original token space.
    """
    if attention.ndim == 4:
        scores = attention.mean(1)[:, label_index, :]
    elif attention.ndim == 3:
        scores = attention[:, label_index, :]
    else:
        raise ValueError("attention must be [B,H,L,N] or [B,L,N]")
    if provenance is not None:
        scores = recover_attribution(scores, provenance)
    return scores


def topk_deletion_mask(scores: torch.Tensor, keep_fraction: float = 0.2) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be [B,N]")
    b, n = scores.shape
    k = max(1, min(n, int(round(n * keep_fraction))))
    idx = torch.topk(scores, k=k, dim=1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    mask.scatter_(1, idx, True)
    return mask