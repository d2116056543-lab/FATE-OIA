from __future__ import annotations

import math
from typing import Iterable

import torch


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().cpu())
    return math.sqrt(total)
