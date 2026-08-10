from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DistributionalRankSketch(nn.Module):
    def __init__(self, num_labels: int = 4, quantiles: int = 16, momentum: float = 0.9) -> None:
        super().__init__()
        self.num_labels, self.quantiles, self.momentum = int(num_labels), int(quantiles), float(momentum)
        self.register_buffer("positive_quantiles", torch.zeros(num_labels, quantiles))
        self.register_buffer("negative_quantiles", torch.zeros(num_labels, quantiles))
        self.register_buffer("positive_count", torch.zeros(num_labels))
        self.register_buffer("negative_count", torch.zeros(num_labels))
        self.register_buffer("last_update", torch.full((num_labels,), -1, dtype=torch.long))

    @torch.no_grad()
    def update(self, scores: Tensor, target: Tensor, update: int) -> None:
        grid = torch.linspace(0, 1, self.quantiles, device=scores.device)
        for label in range(self.num_labels):
            for positive, destination, counter in (
                (True, self.positive_quantiles, self.positive_count),
                (False, self.negative_quantiles, self.negative_count),
            ):
                mask = target[:, label] > 0.5 if positive else target[:, label] <= 0.5
                if not bool(mask.any()):
                    continue
                current = torch.quantile(scores[mask, label].detach().float(), grid).to(destination)
                destination[label].copy_(current if counter[label] == 0 else self.momentum * destination[label] + (1-self.momentum) * current)
                counter[label] += int(mask.sum())
                self.last_update[label] = int(update)

    def loss(self, scores: Tensor, target: Tensor, margin: float = 0.05) -> Tensor:
        losses = []
        for label in range(self.num_labels):
            if self.positive_count[label] <= 0 or self.negative_count[label] <= 0:
                continue
            pos, neg = target[:, label] > 0.5, target[:, label] <= 0.5
            terms = []
            if bool(pos.any()):
                terms.append(F.softplus(self.negative_quantiles[label][None] + margin - scores[pos, label, None]).mean())
            if bool(neg.any()):
                terms.append(F.softplus(scores[neg, label, None] + margin - self.positive_quantiles[label][None]).mean())
            if terms:
                losses.append(torch.stack(terms).mean())
        return torch.stack(losses).mean() if losses else scores.sum() * 0

    def stats(self, update: int) -> dict[str, object]:
        return {"labels_with_positive": int((self.positive_count > 0).sum()),
                "labels_with_negative": int((self.negative_count > 0).sum()),
                "positive_count": self.positive_count.tolist(), "negative_count": self.negative_count.tolist(),
                "age": (int(update) - self.last_update).clamp_min(0).tolist()}


def rank_preservation_loss(final_scores: Tensor, base_scores: Tensor, target: Tensor,
                           rho: float = 0.95, margin: float = 0.05) -> Tensor:
    losses = []
    for label in range(target.shape[1]):
        pos, neg = target[:, label] > 0.5, target[:, label] <= 0.5
        if not bool(pos.any() and neg.any()):
            continue
        base_margin = base_scores[pos, label][:, None] - base_scores[neg, label][None]
        final_margin = final_scores[pos, label][:, None] - final_scores[neg, label][None]
        reliable = base_margin > float(margin)
        if bool(reliable.any()):
            losses.append(F.relu(float(rho) * base_margin.detach() - final_margin)[reliable].mean())
    return torch.stack(losses).mean() if losses else final_scores.sum() * 0


def quantile_rank_preservation_loss(final_scores: Tensor, base_scores: Tensor, target: Tensor,
                                    final_sketch: DistributionalRankSketch,
                                    base_sketch: DistributionalRankSketch,
                                    rho: float = .95, margin: float = .05) -> Tensor:
    """Protect reliable frozen-base margins against the corresponding final-score sketch."""
    losses=[]
    for label in range(target.shape[1]):
        if min(float(base_sketch.positive_count[label]),float(base_sketch.negative_count[label]),
               float(final_sketch.positive_count[label]),float(final_sketch.negative_count[label]))<=0: continue
        pos,neg=target[:,label]>.5,target[:,label]<=.5; terms=[]
        if bool(pos.any()):
            base_margin=base_scores[pos,label,None]-base_sketch.negative_quantiles[label][None]
            final_margin=final_scores[pos,label,None]-final_sketch.negative_quantiles[label][None]
            reliable=base_margin>margin
            if bool(reliable.any()): terms.append(F.relu(rho*base_margin.detach()-final_margin)[reliable].mean())
        if bool(neg.any()):
            base_margin=base_sketch.positive_quantiles[label][None]-base_scores[neg,label,None]
            final_margin=final_sketch.positive_quantiles[label][None]-final_scores[neg,label,None]
            reliable=base_margin>margin
            if bool(reliable.any()): terms.append(F.relu(rho*base_margin.detach()-final_margin)[reliable].mean())
        if terms: losses.append(torch.stack(terms).mean())
    return torch.stack(losses).mean() if losses else final_scores.sum()*0
