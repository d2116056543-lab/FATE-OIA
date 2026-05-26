from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LabelStatistics:
    num_samples: int
    positive_counts: torch.Tensor
    pair_counts: torch.Tensor
    smoothing: float = 1.0


def build_label_statistics(labels: torch.Tensor, smoothing: float = 1.0) -> LabelStatistics:
    """Build multi-label co-occurrence counts from a [N,L] binary label matrix."""
    if labels.ndim != 2:
        raise ValueError("labels must be a [N,L] tensor")
    y = (labels.float() > 0).float()
    return LabelStatistics(
        num_samples=int(y.shape[0]),
        positive_counts=y.sum(0).cpu(),
        pair_counts=(y.t() @ y).cpu(),
        smoothing=float(smoothing),
    )


def conditional_bias_matrix(stats: LabelStatistics, zero_diagonal: bool = True) -> torch.Tensor:
    """Return log P(label_j | label_i) with Laplace smoothing."""
    counts = stats.pair_counts.float() + stats.smoothing
    denom = stats.positive_counts.float().view(-1, 1) + stats.smoothing * counts.shape[1]
    bias = torch.log(counts / denom.clamp_min(1e-8))
    if zero_diagonal:
        bias.fill_diagonal_(0.0)
    return bias


def pmi_bias_matrix(stats: LabelStatistics, clip: float = 3.0, zero_diagonal: bool = True) -> torch.Tensor:
    """Return clipped PMI(label_i,label_j) with Laplace smoothing."""
    l = int(stats.positive_counts.numel())
    n = float(stats.num_samples)
    smooth = float(stats.smoothing)
    pair_prob = (stats.pair_counts.float() + smooth) / (n + smooth * l * l)
    prior = (stats.positive_counts.float() + smooth) / (n + smooth * l)
    bias = torch.log(pair_prob / (prior.view(-1, 1) * prior.view(1, -1)).clamp_min(1e-8))
    bias = bias.clamp(min=-float(clip), max=float(clip))
    if zero_diagonal:
        bias.fill_diagonal_(0.0)
    return bias

