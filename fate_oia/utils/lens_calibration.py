from __future__ import annotations

import torch


def fit_group_shrinkage_threshold(logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor | None = None) -> torch.Tensor:
    """Train-calib-only deterministic threshold fit; returns detached deployment thresholds."""
    candidates = torch.linspace(0.05, 0.95, 19, device=logits.device)
    probs = logits.sigmoid(); best = torch.full((logits.shape[1],), 0.5, device=logits.device)
    for label in range(logits.shape[1]):
        target = labels[:, label].bool(); scores=[]
        for threshold in candidates:
            pred=probs[:, label] >= threshold; tp=(pred & target).sum().float(); fp=(pred & ~target).sum().float(); fn=(~pred & target).sum().float()
            scores.append(2*tp/(2*tp+fp+fn).clamp_min(1.0))
        best[label]=candidates[torch.stack(scores).argmax()]
    if groups is not None:
        for group in groups.unique():
            idx=groups==group; best[idx]=0.5*best[idx]+0.5*best[idx].mean()
    return best.detach()
