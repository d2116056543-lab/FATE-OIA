from __future__ import annotations

import torch
import torch.nn.functional as F


def multilabel_hard_negative_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, margin: float = 0.2, top_k_neg: int = 5) -> torch.Tensor:
    """Pair positive labels against highest-scoring negatives per sample."""
    logits = logits.float()
    labels = labels.float()
    losses = []
    for i in range(logits.shape[0]):
        pos = torch.where(labels[i] > 0)[0]
        neg = torch.where(labels[i] <= 0)[0]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        k = min(int(top_k_neg), int(neg.numel()))
        hard_neg = neg[torch.topk(logits[i, neg], k=k).indices]
        diff = margin - logits[i, pos].view(-1, 1) + logits[i, hard_neg].view(1, -1)
        losses.append(F.relu(diff).mean())
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()
