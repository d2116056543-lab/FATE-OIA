from __future__ import annotations

from contextlib import contextmanager
import torch


class SWALite:
    def __init__(self, top_k: int = 5, start_epoch: int = 5) -> None:
        self.top_k = int(top_k)
        self.start_epoch = int(start_epoch)
        self.snapshots: list[tuple[float, dict[str, torch.Tensor]]] = []

    def consider(self, model, score: float, epoch: int) -> None:
        if epoch < self.start_epoch:
            return
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point and not k.startswith('dino.')}
        self.snapshots.append((float(score), state))
        self.snapshots = sorted(self.snapshots, key=lambda x: x[0], reverse=True)[: self.top_k]

    @property
    def available(self) -> bool:
        return len(self.snapshots) >= 2

    @contextmanager
    def averaged_parameters(self, model):
        if not self.available:
            yield
            return
        keys = self.snapshots[0][1].keys()
        avg = {k: sum(s[k] for _, s in self.snapshots) / len(self.snapshots) for k in keys}
        backup = {}
        state = model.state_dict()
        for k, v in avg.items():
            if k in state:
                backup[k] = state[k].detach().cpu().clone()
                state[k].copy_(v.to(state[k].device, state[k].dtype))
        try:
            yield
        finally:
            state = model.state_dict()
            for k, v in backup.items():
                state[k].copy_(v.to(state[k].device, state[k].dtype))

    def state_dict(self) -> dict:
        return {"top_k": self.top_k, "start_epoch": self.start_epoch, "snapshots": self.snapshots}

    def load_state_dict(self, state: dict) -> None:
        self.top_k = int(state.get("top_k", self.top_k))
        self.start_epoch = int(state.get("start_epoch", self.start_epoch))
        self.snapshots = state.get("snapshots", [])
