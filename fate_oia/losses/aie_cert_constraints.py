from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class AIECertDualState(nn.Module):
    NAMES = ("effect", "necessity", "action_budget", "reason_budget")

    def __init__(self, lr: float = 0.01, ema_decay: float = 0.95, maximum: float = 10.0) -> None:
        super().__init__()
        self.lr = float(lr)
        self.ema_decay = float(ema_decay)
        self.maximum = float(maximum)
        for name in self.NAMES:
            self.register_buffer(f"lambda_{name}", torch.zeros(()))
            self.register_buffer(f"constraint_ema_{name}", torch.zeros(()))

    def primal_loss(self, constraints: dict[str, Tensor]) -> Tensor:
        terms = [getattr(self, f"lambda_{name}") * F.relu(constraints[name]) for name in constraints]
        return torch.stack(terms).sum() if terms else next(self.buffers()).new_zeros(())

    @torch.no_grad()
    def update(self, constraints: dict[str, Tensor], scale: float = 1.0) -> None:
        if not self.training or scale <= 0:
            return
        for name, value in constraints.items():
            ema = getattr(self, f"constraint_ema_{name}")
            lam = getattr(self, f"lambda_{name}")
            ema.mul_(self.ema_decay).add_(value.detach(), alpha=1.0 - self.ema_decay)
            lam.add_(self.lr * scale * ema).clamp_(0.0, self.maximum)
