from __future__ import annotations

import math
from collections.abc import Iterable

import torch


def clip_shared_gradient_budget(parameters: Iterable[torch.nn.Parameter], max_norm: float = 1.0) -> float:
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0
    total_sq = torch.zeros((), device=params[0].grad.device)
    for param in params:
        total_sq = total_sq + param.grad.detach().float().pow(2).sum()
    total_norm = float(total_sq.sqrt().cpu())
    if math.isfinite(total_norm) and total_norm > max_norm > 0:
        scale = float(max_norm / (total_norm + 1e-6))
        for param in params:
            param.grad.mul_(scale)
    return total_norm
