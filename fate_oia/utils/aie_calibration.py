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
    target_prevalence: Tensor | None = None,
    prevalence_multiplier: Tensor | float = 1.0,
    prevalence_support_prior: float = 0.0,
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
    mode = "group_shrinkage"
    prevalence_threshold = None
    if target_prevalence is not None and float(prevalence_support_prior) > 0:
        multiplier = torch.as_tensor(prevalence_multiplier, dtype=logits.dtype).flatten()
        if multiplier.numel() == 1:
            multiplier = multiplier.expand(logits.shape[1])
        if multiplier.numel() != logits.shape[1]:
            raise ValueError("prevalence_multiplier must be scalar or one value per label")
        target_rate = (target_prevalence.float() * multiplier.float()).clamp(1e-4, 0.95)
        probabilities = logits.float().sigmoid()
        prevalence_threshold = torch.stack([
            torch.quantile(probabilities[:, label], 1.0 - target_rate[label])
            for label in range(probabilities.shape[1])
        ])
        trust = support / (support + float(prevalence_support_prior))
        threshold = trust * raw["threshold_prob"].float() + (1 - trust) * prevalence_threshold
        mode = "prevalence_shrinkage"
    return {
        **raw,
        "threshold_prob_raw": raw["threshold_prob"],
        "threshold_prob": threshold,
        "threshold_logit": torch.logit(threshold.clamp(1e-5, 1 - 1e-5)),
        "calibration_mode": mode,
        "prevalence_threshold_prob": prevalence_threshold,
    }


def apply_posthoc_threshold(logits: Tensor, threshold_prob: Tensor) -> Tensor:
    return logits.float() - torch.logit(threshold_prob.to(logits.device).float().clamp(1e-5, 1 - 1e-5))


