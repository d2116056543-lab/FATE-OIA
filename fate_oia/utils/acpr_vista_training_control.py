from __future__ import annotations

import math


def warmup_cosine_by_update(step: int, total_steps: int, warmup_steps: int, min_lr_ratio: float = 0.05) -> float:
    if step < max(warmup_steps, 1):
        return max(float(step + 1) / float(max(warmup_steps, 1)), min_lr_ratio)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def vista_scale_for_epoch(epoch: int) -> float:
    if epoch < 3:
        return 0.05
    if epoch < 9:
        return 0.15
    return 0.08

