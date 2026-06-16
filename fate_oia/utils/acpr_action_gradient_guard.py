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
        self.last_stats = {
            "grad_dot_action_aux": 0.0,
            "grad_cos_action_aux": 0.0,
            "grad_conflict": False,
            "grad_conflict_rate_window": 0.0,
            "aux_projected_norm": 0.0,
            "action_grad_norm": 0.0,
            "mode": self.mode,
            "active": False,
        }

    def should_run(self, epoch: int, step: int) -> bool:
        return self.mode != "disabled" and epoch >= self.project_after_epoch and step % max(1, self.every_n_steps) == 0

    @staticmethod
    def project(conflict_grad: torch.Tensor, action_grad: torch.Tensor) -> torch.Tensor:
        dot = torch.dot(conflict_grad.flatten(), action_grad.flatten())
        denom = torch.dot(action_grad.flatten(), action_grad.flatten()).clamp_min(1e-12)
        if dot < 0:
            return conflict_grad - dot / denom * action_grad
        return conflict_grad

    def capture_action_grads(self, action_loss: torch.Tensor, params: list[torch.nn.Parameter]) -> list[torch.Tensor | None]:
        return list(torch.autograd.grad(action_loss, params, retain_graph=True, allow_unused=True))

    def project_model_grads(self, named_params: list[tuple[str, torch.nn.Parameter]], action_grads: list[torch.Tensor | None], epoch: int, step: int) -> dict:
        total_dot = 0.0
        total_action_norm = 0.0
        total_grad_norm = 0.0
        projected_norm = 0.0
        conflicts = 0
        checked = 0
        for (_, param), action_grad in zip(named_params, action_grads):
            if param.grad is None or action_grad is None:
                continue
            g = param.grad.detach()
            a = action_grad.detach().to(g.device, g.dtype)
            dot = torch.dot(g.flatten(), a.flatten())
            action_norm = torch.linalg.vector_norm(a)
            grad_norm = torch.linalg.vector_norm(g)
            total_dot += float(dot.detach().cpu())
            total_action_norm += float(action_norm.detach().cpu())
            total_grad_norm += float(grad_norm.detach().cpu())
            checked += 1
            if dot < 0:
                conflicts += 1
                if self.mode in {"project", "log_then_project"}:
                    new_g = self.project(g, a)
                    projected_norm += float(torch.linalg.vector_norm(g - new_g).detach().cpu())
                    param.grad.copy_(new_g)
        conflict_rate = float(conflicts) / max(float(checked), 1.0)
        self.window.append(conflict_rate)
        self.window = self.window[-50:]
        cos = total_dot / max(total_action_norm * total_grad_norm, 1e-12)
        self.last_stats = {
            "epoch": epoch,
            "step": step,
            "grad_dot_action_aux": total_dot,
            "grad_cos_action_aux": cos,
            "grad_conflict": bool(conflicts > 0),
            "grad_conflict_rate_window": float(sum(self.window) / max(len(self.window), 1)),
            "aux_projected_norm": projected_norm,
            "action_grad_norm": total_action_norm,
            "mode": self.mode,
            "active": True,
            "checked_param_count": checked,
            "conflict_param_count": conflicts,
        }
        return dict(self.last_stats)

    def stats(self, epoch: int, step: int, grad_norm: float = 0.0, projected_norm: float = 0.0) -> dict:
        if self.last_stats.get("epoch") == epoch and self.last_stats.get("step") == step:
            return dict(self.last_stats)
        out = dict(self.last_stats)
        out.update({"epoch": epoch, "step": step})
        return out

