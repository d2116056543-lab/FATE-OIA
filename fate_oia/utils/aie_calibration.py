from __future__ import annotations

import torch
from torch import Tensor

from .acpr_threshold_search import search_best_thresholds_for_f1


def fit_posthoc_thresholds(
    logits: Tensor,
    targets: Tensor,
    groups: list[list[int]],
    shrinkage_support: float = 50.0,
    grid_step: float = 0.01,
) -> dict[str, Tensor]:
    grid = torch.arange(0.01, 0.95001, float(grid_step))
    raw = search_best_thresholds_for_f1(logits, targets, grid=grid)
    threshold = raw["threshold_prob"].clone()
    support = raw["support_pos"].float()
    for indices in groups:
        if not indices:
            continue
        group_threshold = threshold[indices].median()
        lam = support[indices] / (support[indices] + float(shrinkage_support))
        threshold[indices] = lam * threshold[indices] + (1 - lam) * group_threshold
    return {**raw, "threshold_prob_raw": raw["threshold_prob"], "threshold_prob": threshold, "threshold_logit": torch.logit(threshold.clamp(1e-5, 1 - 1e-5))}


def apply_posthoc_threshold(logits: Tensor, threshold_prob: Tensor) -> Tensor:
    return logits.float() - torch.logit(threshold_prob.to(logits.device).float().clamp(1e-5, 1 - 1e-5))


