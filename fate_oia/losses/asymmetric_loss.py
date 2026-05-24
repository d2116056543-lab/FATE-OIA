from __future__ import annotations

import torch
from torch import nn


def asymmetric_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Asymmetric multi-label loss for long-tailed BDD-OIA labels."""
    targets = targets.float()
    probs_pos = torch.sigmoid(logits.float())
    probs_neg = 1.0 - probs_pos
    if clip and clip > 0:
        probs_neg = (probs_neg + clip).clamp(max=1.0)
    loss_pos = targets * torch.log(probs_pos.clamp_min(eps))
    loss_neg = (1.0 - targets) * torch.log(probs_neg.clamp_min(eps))
    if gamma_pos > 0 or gamma_neg > 0:
        pt = probs_pos * targets + probs_neg * (1.0 - targets)
        gamma = gamma_pos * targets + gamma_neg * (1.0 - targets)
        weight = torch.pow((1.0 - pt).clamp_min(eps), gamma)
        loss_pos = loss_pos * weight
        loss_neg = loss_neg * weight
    loss = -(loss_pos + loss_neg)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


class AsymmetricLossMultiLabel(nn.Module):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, clip: float = 0.05, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return asymmetric_loss_with_logits(
            logits,
            targets,
            gamma_pos=self.gamma_pos,
            gamma_neg=self.gamma_neg,
            clip=self.clip,
            reduction=self.reduction,
        )