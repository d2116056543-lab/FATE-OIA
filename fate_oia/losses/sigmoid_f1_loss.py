from __future__ import annotations

import torch


def sigmoid_macro_f1_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits.float())
    y = targets.float()
    tp = (probs * y).sum(dim=0)
    fp = (probs * (1.0 - y)).sum(dim=0)
    fn = ((1.0 - probs) * y).sum(dim=0)
    soft_f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return 1.0 - soft_f1.mean()
