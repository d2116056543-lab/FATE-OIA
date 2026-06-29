from __future__ import annotations

import torch


class PMCalConflictAwareOptimizer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        enabled: bool = True,
        shared_params: list[torch.nn.Parameter] | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.enabled = bool(enabled)
        self.shared_params = [p for p in (shared_params or []) if p.requires_grad]
        self.last_stats = {"projection_applied_count": 0, "grad_cosine": 0.0}

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def step(self) -> None:
        self.optimizer.step()

    def step_losses(self, loss_groups: dict[str, torch.Tensor]) -> dict[str, float]:
        if not self.enabled or len(loss_groups) <= 1 or not self.shared_params:
            total = sum(loss_groups.values())
            total.backward()
            self.optimizer.step()
            self.last_stats = {"projection_applied_count": 0, "grad_cosine": 0.0}
            return self.last_stats
        names = list(loss_groups)
        grads: list[torch.Tensor] = []
        for name in names:
            grad_parts = torch.autograd.grad(
                loss_groups[name],
                self.shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            flat = []
            for param, grad in zip(self.shared_params, grad_parts):
                flat.append(torch.zeros_like(param).reshape(-1) if grad is None else grad.reshape(-1))
            grads.append(torch.cat(flat))
        projected = [g.clone() for g in grads]
        projection_count = 0
        cos_values: list[torch.Tensor] = []
        for i in range(len(projected)):
            for j in range(len(projected)):
                if i == j:
                    continue
                denom = projected[i].norm() * grads[j].norm()
                if denom <= 0:
                    continue
                cosine = torch.dot(projected[i], grads[j]) / denom.clamp_min(1e-12)
                cos_values.append(cosine.detach())
                if cosine < 0:
                    projected[i] = projected[i] - torch.dot(projected[i], grads[j]) / grads[j].dot(grads[j]).clamp_min(1e-12) * grads[j]
                    projection_count += 1
        merged = torch.stack(projected, 0).sum(0)
        offset = 0
        for param in self.shared_params:
            n = param.numel()
            param.grad = merged[offset : offset + n].view_as(param).clone()
            offset += n
        # Non-shared parameters still receive the ordinary summed objective gradient.
        non_shared = {id(p) for p in self.shared_params}
        other_params = [p for group in self.optimizer.param_groups for p in group["params"] if p.requires_grad and id(p) not in non_shared]
        if other_params:
            other_grads = torch.autograd.grad(sum(loss_groups.values()), other_params, allow_unused=True)
            for param, grad in zip(other_params, other_grads):
                if grad is not None:
                    param.grad = grad.clone()
        self.optimizer.step()
        mean_cos = torch.stack(cos_values).mean().item() if cos_values else 0.0
        self.last_stats = {"projection_applied_count": int(projection_count), "grad_cosine": float(mean_cos)}
        return self.last_stats
