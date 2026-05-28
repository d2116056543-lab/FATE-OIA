from __future__ import annotations

import torch
import torch.nn.functional as F


def reason_pairwise_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    label_indices: list[int] | tuple[int, ...] | None = None,
    margin: float = 0.2,
    max_pairs_per_label: int = 256,
) -> torch.Tensor:
    if logits.shape != targets.shape:
        raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    indices = list(range(logits.shape[1])) if label_indices is None else [int(x) for x in label_indices]
    losses: list[torch.Tensor] = []
    for idx in indices:
        scores = logits[:, idx].float()
        y = targets[:, idx].float()
        pos = scores[y > 0.5]
        neg = scores[y <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        pos = pos[:max_pairs_per_label]
        neg = neg[:max_pairs_per_label]
        losses.append(F.softplus(float(margin) - (pos[:, None] - neg[None, :])).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()
