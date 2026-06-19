from __future__ import annotations


class PACETrainingControl:
    def __init__(self, min_epoch: int = 8, patience: int = 2, min_delta: float = 1e-4, non_threshold_lr_multiplier: float = 0.20, threshold_lr_multiplier: float = 0.50) -> None:
        self.min_epoch = int(min_epoch)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.non_threshold_lr_multiplier = float(non_threshold_lr_multiplier)
        self.threshold_lr_multiplier = float(threshold_lr_multiplier)
        self.best = float("-inf")
        self.best_epoch = -1
        self.bad_epochs = 0
        self.applied = False

    def update(self, epoch: int, train_calib_joint: float) -> dict[str, float | int | bool]:
        accepted = False
        if train_calib_joint > self.best + self.min_delta:
            self.best = float(train_calib_joint)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            accepted = True
        elif epoch >= self.min_epoch:
            self.bad_epochs += 1
        apply = (not self.applied) and epoch >= self.min_epoch and self.bad_epochs >= self.patience
        if apply:
            self.applied = True
        return {
            "teacher_accepted": accepted,
            "cooldown_apply": apply,
            "cooldown_applied": self.applied,
            "teacher_best_epoch": self.best_epoch,
            "teacher_best_joint": self.best,
            "cooldown_bad_epochs": self.bad_epochs,
        }
