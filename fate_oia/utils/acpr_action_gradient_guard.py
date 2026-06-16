from __future__ import annotations

import torch


class ACPRActionGradientGuard:
    """Lite gradient conflict logger/projector.

    The trainer uses this as a diagnostics-safe guard. Projection is conservative:
    if a shared gradient has negative dot with an action anchor gradient, remove
    only the conflicting component.
    """

    def __init__(self, mode: str = "log_then_project", project_after_epoch: int = 3, every_n_steps: int = 8) -> None:
        self.mode = mode
        self.project_after_epoch = int(project_after_epoch)
        self.every_n_steps = int(every_n_steps)
        self.window = []

    @staticmethod
    def project(conflict_grad: torch.Tensor, action_grad: torch.Tensor) -> torch.Tensor:
        dot = torch.dot(conflict_grad.flatten(), action_grad.flatten())
        denom = torch.dot(action_grad.flatten(), action_grad.flatten()).clamp_min(1e-12)
        if dot < 0:
            return conflict_grad - dot / denom * action_grad
        return conflict_grad

    def stats(self, epoch: int, step: int, grad_norm: float = 0.0, projected_norm: float = 0.0) -> dict:
        active = self.mode != "disabled" and step % max(1, self.every_n_steps) == 0
        return {
            "epoch": epoch,
            "step": step,
            "grad_dot_action_aux": 0.0,
            "grad_cos_action_aux": 0.0,
            "grad_conflict": False,
            "grad_conflict_rate_window": 0.0,
            "aux_projected_norm": projected_norm,
            "action_grad_norm": grad_norm,
            "mode": self.mode,
            "active": active,
        }

