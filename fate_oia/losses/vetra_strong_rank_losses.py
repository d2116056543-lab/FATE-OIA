from __future__ import annotations

import torch
from torch import Tensor


def _zero(logits: Tensor) -> Tensor:
    return logits.sum() * 0.0


def action_pairwise_ap_loss(logits: Tensor, targets: Tensor, temperature: float = 0.25) -> Tensor:
    """Label-wise positive-vs-negative logistic ranking surrogate."""
    losses = []
    for label in range(targets.shape[1]):
        positive = logits[targets[:, label] > 0.5, label]
        negative = logits[targets[:, label] <= 0.5, label]
        if positive.numel() and negative.numel():
            margin = positive[:, None] - negative[None, :]
            losses.append(torch.nn.functional.softplus(-margin / float(temperature)).mean())
    return torch.stack(losses).mean() if losses else _zero(logits)


def action_smooth_ap_loss(logits: Tensor, targets: Tensor, temperature: float = 0.10) -> Tensor:
    """Smooth label-wise AP surrogate over the effective optimizer batch."""
    losses = []
    for label in range(targets.shape[1]):
        score = logits[:, label]
        positive = score[targets[:, label] > 0.5]
        if not positive.numel() or positive.numel() == score.numel():
            continue
        rank = 1.0 + torch.sigmoid((score[None, :] - positive[:, None]) / float(temperature)).sum(-1)
        positive_rank = 1.0 + torch.sigmoid(
            (positive[None, :] - positive[:, None]) / float(temperature)
        ).sum(-1)
        losses.append(1.0 - (positive_rank / rank.clamp_min(1.0)).mean())
    return torch.stack(losses).mean() if losses else _zero(logits)


def base_margin_trust_loss(
    logits: Tensor,
    base_logits: Tensor,
    targets: Tensor,
    tolerance: float = 0.02,
) -> Tensor:
    """Prevent the refiner from erasing already-correct base ranking margins."""
    losses = []
    for label in range(targets.shape[1]):
        positive = targets[:, label] > 0.5
        negative = ~positive
        if positive.any() and negative.any():
            base_margin = base_logits[positive, label][:, None] - base_logits[negative, label][None, :]
            refined_margin = logits[positive, label][:, None] - logits[negative, label][None, :]
            protected = base_margin > 0
            if protected.any():
                losses.append(
                    torch.relu(base_margin.detach() - float(tolerance) - refined_margin)[protected].mean()
                )
    return torch.stack(losses).mean() if losses else _zero(logits)


def residual_energy_loss(delta: Tensor) -> Tensor:
    return delta.square().mean()
