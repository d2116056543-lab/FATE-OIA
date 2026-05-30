from __future__ import annotations

import torch
import torch.nn.functional as F


def hard_logit_pairwise_ranking_loss(logits: torch.Tensor, targets: torch.Tensor, tail_labels: tuple[int, ...] = ()) -> torch.Tensor:
    idx = [i for i in tail_labels if 0 <= i < logits.shape[1]]
    if not idx:
        idx = list(range(logits.shape[1]))
    losses = []
    for i in idx:
        pos = logits[:, i][targets[:, i] > 0.5]
        neg = logits[:, i][targets[:, i] < 0.5]
        if pos.numel() and neg.numel():
            hard_neg = neg.topk(min(16, neg.numel())).values
            losses.append(F.softplus(0.2 - pos.mean() + hard_neg.mean()))
    return torch.stack(losses).mean() if losses else logits.new_zeros(())


def tail_causal_effect_ranking_loss(direct_effect: torch.Tensor, targets: torch.Tensor, tail_labels: tuple[int, ...] = ()) -> torch.Tensor:
    idx = [i for i in tail_labels if 0 <= i < direct_effect.shape[1]]
    if not idx:
        return direct_effect.new_zeros(())
    losses = []
    for i in idx:
        pos = direct_effect[:, i][targets[:, i] > 0.5]
        neg = direct_effect[:, i][targets[:, i] < 0.5]
        if pos.numel() and neg.numel():
            losses.append(F.softplus(0.1 - pos.mean() + neg.mean()))
    return torch.stack(losses).mean() if losses else direct_effect.new_zeros(())


def sigmoid_macro_f1_surrogate(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    y = targets.float()
    tp = (probs * y).sum(0)
    fp = (probs * (1.0 - y)).sum(0)
    fn = ((1.0 - probs) * y).sum(0)
    f1 = 2.0 * tp / (2.0 * tp + fp + fn + 1e-6)
    return 1.0 - f1.mean()

