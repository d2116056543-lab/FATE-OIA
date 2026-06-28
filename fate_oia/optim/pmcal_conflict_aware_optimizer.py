from __future__ import annotations

import torch


class PMCalConflictAwareOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer, enabled: bool = True) -> None:
        self.optimizer = optimizer
        self.enabled = bool(enabled)
        self.last_stats = {"projection_applied_count": 0, "grad_cosine": 0.0}

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.optimizer.step()

    def step_losses(self, loss_groups: dict[str, torch.Tensor]) -> dict[str, float]:
        total = sum(loss_groups.values())
        total.backward()
        self.optimizer.step()
        self.last_stats = {"projection_applied_count": 0, "grad_cosine": 0.0}
        return self.last_stats
