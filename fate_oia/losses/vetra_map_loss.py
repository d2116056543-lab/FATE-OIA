from __future__ import annotations

import torch
from torch import Tensor, nn


class VETRAMAPLoss(nn.Module):
    """Clean-room smooth AP surrogate with only per-label scalar EMA state."""

    def __init__(self, action_dim: int = 4, temperature: float = .10, momentum: float = .9) -> None:
        super().__init__()
        self.temperature, self.momentum = float(temperature), float(momentum)
        self.register_buffer("positive_ema", torch.zeros(action_dim))
        self.register_buffer("negative_ema", torch.zeros(action_dim))
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        losses = []
        for action in range(targets.shape[1]):
            score, target = logits[:, action], targets[:, action] > .5
            positive, negative = score[target], score[~target]
            if positive.numel() == 0 or negative.numel() == 0:
                continue
            # Approximate each positive's rank and positive rank with pairwise sigmoids.
            all_score = score[None]
            rank = 1 + torch.sigmoid((all_score - positive[:, None]) / self.temperature).sum(-1)
            positive_rank = 1 + torch.sigmoid((positive[None] - positive[:, None]) / self.temperature).sum(-1)
            losses.append(1 - (positive_rank / rank.clamp_min(1)).mean())
            if self.training:
                with torch.no_grad():
                    self.positive_ema[action].lerp_(positive.new_tensor(float(positive.numel())), 1 - self.momentum)
                    self.negative_ema[action].lerp_(negative.new_tensor(float(negative.numel())), 1 - self.momentum)
        if self.training:
            self.updates.add_(1)
        return torch.stack(losses).mean() if losses else logits.sum() * 0
