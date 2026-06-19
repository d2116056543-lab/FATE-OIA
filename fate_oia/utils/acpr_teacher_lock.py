from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class TeacherLockState:
    best_joint: float = float("-inf")
    best_action: float = float("-inf")
    best_exp: float = float("-inf")
    best_epoch: int = -1

    def update(self, epoch: int, joint: float, action: float, exp: float, min_delta: float = 1e-4, action_tolerance: float = 1e-3, exp_tolerance: float = 1e-3) -> bool:
        if joint > self.best_joint + min_delta and action >= self.best_action - action_tolerance and exp >= self.best_exp - exp_tolerance:
            self.best_joint = float(joint)
            self.best_action = float(action)
            self.best_exp = float(exp)
            self.best_epoch = int(epoch)
            return True
        return False

    def to_dict(self) -> dict:
        return asdict(self)
