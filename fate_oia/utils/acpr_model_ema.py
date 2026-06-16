from __future__ import annotations

from contextlib import contextmanager
import torch


class ModelEMA:
    def __init__(self, model, decay: float = 0.995, start_epoch: int = 3) -> None:
        self.decay = float(decay)
        self.start_epoch = int(start_epoch)
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.num_updates = 0

    def update(self, model, epoch: int) -> None:
        if epoch < self.start_epoch:
            return
        state = model.state_dict()
        for k, v in state.items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().cpu(), alpha=1.0 - self.decay)
        self.num_updates += 1

    @contextmanager
    def average_parameters(self, model):
        backup = {}
        state = model.state_dict()
        for k, avg in self.shadow.items():
            if k in state:
                backup[k] = state[k].detach().cpu().clone()
                state[k].copy_(avg.to(state[k].device, state[k].dtype))
        try:
            yield
        finally:
            state = model.state_dict()
            for k, v in backup.items():
                state[k].copy_(v.to(state[k].device, state[k].dtype))

    @property
    def available(self) -> bool:
        return self.num_updates > 0

    def state_dict(self) -> dict:
        return {"decay": self.decay, "start_epoch": self.start_epoch, "shadow": self.shadow, "num_updates": self.num_updates}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.start_epoch = int(state.get("start_epoch", self.start_epoch))
        self.shadow = {k: v.clone().cpu() for k, v in state.get("shadow", {}).items()}
        self.num_updates = int(state.get("num_updates", 0))

