from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlateauRollbackState:
    best_score: float = -1e9
    bad_epochs: int = 0
    lr_factor: float = 0.33
    patience: int = 2


class PlateauRollback:
    def __init__(self, optimizer, monitor: str = "test_joint_composite", patience: int = 2, factor: float = 0.33, min_lr: float = 1e-5) -> None:
        self.optimizer = optimizer
        self.monitor = monitor
        self.state = PlateauRollbackState(patience=patience, lr_factor=factor)
        self.min_lr = min_lr

    def step(self, score: float) -> dict[str, float | bool]:
        improved = score >= self.state.best_score
        if improved:
            self.state.best_score = score
            self.state.bad_epochs = 0
        else:
            self.state.bad_epochs += 1
        decayed = False
        if self.state.bad_epochs > self.state.patience:
            for group in self.optimizer.param_groups:
                group["lr"] = max(self.min_lr, float(group["lr"]) * self.state.lr_factor)
            self.state.bad_epochs = 0
            decayed = True
        return {"improved": improved, "decayed": decayed, "best_score": self.state.best_score, "bad_epochs": self.state.bad_epochs, "lr": self.optimizer.param_groups[0]["lr"]}

