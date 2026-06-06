from __future__ import annotations

import torch
import torch.nn.functional as F


def selected_vs_random_margin_loss(
    selected_drop: torch.Tensor,
    random_drop: torch.Tensor,
    *,
    positive_mask: torch.Tensor | None = None,
    margin: float = 0.05,
) -> torch.Tensor:
    """Penalize evidence when selected deletion is no stronger than random.

    Good evidence should produce a larger loss/logit drop when selected evidence
    is removed than when random evidence of equal size is removed.
    """

    loss = F.relu(float(margin) + random_drop - selected_drop)
    if positive_mask is not None:
        mask = positive_mask.to(loss.device, loss.dtype)
        denom = mask.sum().clamp_min(1.0)
        return (loss * mask).sum() / denom
    return loss.mean()
