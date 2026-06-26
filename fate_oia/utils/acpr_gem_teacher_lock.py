from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TeacherBestLock:
    min_delta: float = 1e-4
    action_tolerance: float = 0.001
    exp_tolerance: float = 0.001
    best_joint: float = float("-inf")
    best_action: float = float("-inf")
    best_exp: float = float("-inf")
    best_epoch: int = -1
    best_theta: torch.Tensor | None = None

    def maybe_accept(self, theta: torch.Tensor, joint: float, action: float, exp: float, epoch: int) -> dict:
        accepted = (
            float(joint) > self.best_joint + self.min_delta
            and float(action) >= self.best_action - self.action_tolerance
            and float(exp) >= self.best_exp - self.exp_tolerance
        )
        if self.best_theta is None:
            accepted = True
        if accepted:
            self.best_theta = theta.detach().clone()
            self.best_joint = float(joint)
            self.best_action = float(action)
            self.best_exp = float(exp)
            self.best_epoch = int(epoch)
        return {
            "accepted": bool(accepted),
            "best_epoch": self.best_epoch,
            "best_joint": self.best_joint,
            "best_action": self.best_action,
            "best_exp": self.best_exp,
        }
