from __future__ import annotations

import torch
import torch.nn.functional as F


def tail_margin_ranking_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    tail_indices: list[int] | tuple[int, ...],
    margin: float = 0.2,
    max_pairs_per_label: int = 256,
) -> torch.Tensor:
    """Pairwise softplus ranking loss for weak/tail multi-label reasons.

    For each selected label, positives should score above negatives by
    ``margin``. The implementation is deterministic and bounded so it stays
    cheap for full test-split cached-logit diagnostics.
    """

    if logits.shape != targets.shape:
        raise ValueError(f"logits/targets shape mismatch: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    losses: list[torch.Tensor] = []
    for label_idx in tail_indices:
        scores = logits[:, int(label_idx)].float()
        y = targets[:, int(label_idx)].float()
        pos = scores[y > 0.5]
        neg = scores[y <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        if pos.numel() > max_pairs_per_label:
            pos = pos[:max_pairs_per_label]
        if neg.numel() > max_pairs_per_label:
            neg = neg[:max_pairs_per_label]
        pair_margin = float(margin) - (pos.view(-1, 1) - neg.view(1, -1))
        losses.append(F.softplus(pair_margin).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()
