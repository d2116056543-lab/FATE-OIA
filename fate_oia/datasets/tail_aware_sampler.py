from __future__ import annotations

from typing import Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def compute_tail_aware_sample_weights(samples: Sequence, reason_dim: int = 21, power: float = 0.5) -> torch.Tensor:
    if not samples:
        return torch.empty(0, dtype=torch.float32)
    labels = torch.tensor([getattr(sample, "reason") for sample in samples], dtype=torch.float32)
    if labels.ndim != 2 or labels.shape[1] != reason_dim:
        raise ValueError(f"Expected reason labels [N,{reason_dim}], got {tuple(labels.shape)}")
    counts = labels.sum(dim=0).clamp_min(1.0)
    inv = counts.pow(-float(power))
    weights = (labels * inv.view(1, -1)).sum(dim=1)
    weights = weights + 1.0 / float(len(samples))
    return weights.float()


def build_tail_aware_sampler(dataset, reason_dim: int = 21, power: float = 0.5) -> WeightedRandomSampler:
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise ValueError("tail-aware sampler expects a dataset with .samples")
    weights = compute_tail_aware_sample_weights(samples, reason_dim=reason_dim, power=power)
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
