from __future__ import annotations

import torch


def finite_or_zero(loss: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isfinite(loss), loss, torch.zeros_like(loss))


def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += float(p.grad.detach().norm().cpu().item()) ** 2
    return total ** 0.5
