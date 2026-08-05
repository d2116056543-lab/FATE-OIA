from __future__ import annotations

import torch
import torch.nn.functional as F

from fate_oia.losses.asymmetric_loss import asymmetric_loss_with_logits


def action_asl_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return asymmetric_loss_with_logits(logits, targets.float(), gamma_neg=4, gamma_pos=0, clip=0.05)


def action_soft_f1_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = logits.sigmoid(); target = targets.float()
    tp = (probs * target).sum(0); fp = (probs * (1 - target)).sum(0); fn = ((1 - probs) * target).sum(0)
    return 1.0 - ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean()


def action_cardinality_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(logits.sigmoid().sum(-1), targets.float().sum(-1))
