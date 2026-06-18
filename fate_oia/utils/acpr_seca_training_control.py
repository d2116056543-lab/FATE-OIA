from __future__ import annotations

import math


def update_warmup_cosine_multiplier(update_idx: int, total_updates: int, warmup_updates: int, min_lr_ratio: float = 0.05) -> float:
    if update_idx < warmup_updates:
        return max(float(update_idx + 1) / max(int(warmup_updates), 1), float(min_lr_ratio))
    progress = (update_idx - warmup_updates) / max(total_updates - warmup_updates, 1)
    return float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * 0.5 * (1.0 + math.cos(math.pi * progress))


class ACPRSECATrainingControl:
    def __init__(self, cooldown_start_epoch: int = 6, patience: int = 2, non_threshold_lr_mult: float = 0.20, threshold_lr_mult: float = 0.50) -> None:
        self.cooldown_start_epoch = int(cooldown_start_epoch)
        self.patience = int(patience)
        self.non_threshold_lr_mult = float(non_threshold_lr_mult)
        self.threshold_lr_mult = float(threshold_lr_mult)
        self.best = float("-inf")
        self.bad_epochs = 0
        self.cooldown_applied = False

    def update(self, epoch: int, train_calib_metric: float) -> dict:
        metric = float(train_calib_metric)
        if metric > self.best + 1e-12:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        apply = (epoch >= self.cooldown_start_epoch and self.bad_epochs >= self.patience and not self.cooldown_applied)
        if apply:
            self.cooldown_applied = True
        return {
            "cooldown_active": bool(self.cooldown_applied),
            "cooldown_newly_applied": bool(apply),
            "bad_epochs": int(self.bad_epochs),
            "best_train_calib_metric": float(self.best),
            "non_threshold_lr_mult": self.non_threshold_lr_mult if self.cooldown_applied else 1.0,
            "threshold_lr_mult": self.threshold_lr_mult if self.cooldown_applied else 1.0,
        }


def apply_lr_cooldown(optimizer, state: dict, lr_multiplier: float = 1.0) -> None:
    for group in optimizer.param_groups:
        base = group.setdefault("base_lr", group["lr"])
        if group.get("name") == "threshold":
            group["lr"] = base * float(lr_multiplier) * float(state.get("threshold_lr_mult", 1.0))
        else:
            group["lr"] = base * float(lr_multiplier) * float(state.get("non_threshold_lr_mult", 1.0))
