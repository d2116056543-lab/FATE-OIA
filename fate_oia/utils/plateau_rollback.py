from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from fate_oia.utils.cafe_artifacts import append_jsonl


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


@dataclass
class PlateauRestoreState:
    best_score: float = -1e9
    bad_epochs: int = 0
    restore_count: int = 0


class PlateauRestore:
    """Plateau scheduler that restores the best model/optimizer before LR decay."""

    def __init__(self, patience: int = 2, factor: float = 0.33, min_lr: float = 1e-5, max_restores: int = 3) -> None:
        self.patience = int(patience)
        self.factor = float(factor)
        self.min_lr = float(min_lr)
        self.max_restores = int(max_restores)
        self.state = PlateauRestoreState()
        self._best_model: dict[str, Any] | None = None
        self._best_optimizer: dict[str, Any] | None = None

    def save_best(self, epoch: int, score: float, model, optimizer, output_dir: str | Path) -> None:
        self.state.best_score = float(score)
        self.state.bad_epochs = 0
        self._best_model = deepcopy(model.state_dict())
        self._best_optimizer = deepcopy(optimizer.state_dict())
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "score": score, "model": self._best_model, "optimizer": self._best_optimizer}, out / "checkpoint_best_restore.pth")

    def restore_best_and_decay_lr(self, epoch: int, model, optimizer, output_dir: str | Path) -> dict[str, Any]:
        restored = False
        if self._best_model is not None and self._best_optimizer is not None and self.state.restore_count < self.max_restores:
            model.load_state_dict(self._best_model)
            optimizer.load_state_dict(self._best_optimizer)
            restored = True
            self.state.restore_count += 1
        for group in optimizer.param_groups:
            group["lr"] = max(self.min_lr, float(group["lr"]) * self.factor)
        self.state.bad_epochs = 0
        event = {
            "epoch": epoch,
            "restored": restored,
            "restore_count": self.state.restore_count,
            "best_score": self.state.best_score,
            "lr": optimizer.param_groups[0]["lr"],
        }
        append_jsonl(Path(output_dir) / "plateau_restore_event.jsonl", event)
        return event

    def step(self, score: float, epoch: int, model, optimizer, scheduler_state: dict[str, Any] | None, output_dir: str | Path) -> dict[str, Any]:
        improved = float(score) >= self.state.best_score
        if improved:
            self.save_best(epoch, float(score), model, optimizer, output_dir)
            return {"improved": True, "restored": False, "decayed": False, "best_score": self.state.best_score, "bad_epochs": 0, "lr": optimizer.param_groups[0]["lr"]}
        self.state.bad_epochs += 1
        if self.state.bad_epochs > self.patience:
            event = self.restore_best_and_decay_lr(epoch, model, optimizer, output_dir)
            return {"improved": False, "restored": bool(event["restored"]), "decayed": True, "best_score": self.state.best_score, "bad_epochs": 0, "lr": optimizer.param_groups[0]["lr"]}
        return {"improved": False, "restored": False, "decayed": False, "best_score": self.state.best_score, "bad_epochs": self.state.bad_epochs, "lr": optimizer.param_groups[0]["lr"]}

    def state_dict(self) -> dict[str, Any]:
        return {
            "state": vars(self.state),
            "best_model": self._best_model,
            "best_optimizer": self._best_optimizer,
            "patience": self.patience,
            "factor": self.factor,
            "min_lr": self.min_lr,
            "max_restores": self.max_restores,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raw = state.get("state", {})
        self.state = PlateauRestoreState(**{k: raw.get(k, getattr(PlateauRestoreState(), k)) for k in ("best_score", "bad_epochs", "restore_count")})
        self._best_model = state.get("best_model")
        self._best_optimizer = state.get("best_optimizer")
