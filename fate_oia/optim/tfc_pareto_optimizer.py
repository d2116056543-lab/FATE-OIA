from __future__ import annotations

import torch


class TFCParetoOptimizer:
    """Action-priority PCGrad helper.

    The training code may use structural firewall instead of full PCGrad for
    speed, but this class is functional and logs projection evidence.
    """

    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = [p for p in params if p.requires_grad]

    @staticmethod
    def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.dot(a, b) / (a.norm() * b.norm()).clamp_min(1e-12)

    def flatten_grads(self) -> torch.Tensor:
        chunks = []
        for p in self.params:
            if p.grad is not None:
                chunks.append(p.grad.detach().flatten())
        if not chunks:
            return torch.zeros(1)
        return torch.cat(chunks)

    @staticmethod
    def project_away_from_action(action_grad: torch.Tensor, other_grad: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Project a conflicting auxiliary gradient away from action gradient."""
        if action_grad.numel() != other_grad.numel():
            raise ValueError("action_grad and other_grad must have the same flattened size")
        dot = torch.dot(action_grad, other_grad)
        denom = torch.dot(action_grad, action_grad).clamp_min(1e-12)
        cosine = TFCParetoOptimizer.cosine(action_grad, other_grad)
        projected = other_grad
        did_project = bool(dot.detach().cpu() < 0)
        if did_project:
            projected = other_grad - dot / denom * action_grad
        return projected, {
            "cosine": float(cosine.detach().cpu()),
            "dot": float(dot.detach().cpu()),
            "projected": did_project,
        }

    @staticmethod
    def combine_action_priority(action_grad: torch.Tensor, other_grads: list[torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """Combine gradients with action-priority PCGrad semantics."""
        combined = action_grad.clone()
        projection_count = 0
        cosines: list[float] = []
        for other in other_grads:
            projected, stats = TFCParetoOptimizer.project_away_from_action(action_grad, other)
            combined = combined + projected
            projection_count += int(stats["projected"])
            cosines.append(float(stats["cosine"]))
        return combined, {
            "enabled": True,
            "projection_count": projection_count,
            "cosines": cosines,
        }

    def assign_flat_grad(self, flat_grad: torch.Tensor) -> None:
        """Write a flattened gradient vector back to the tracked parameters."""
        offset = 0
        for param in self.params:
            n = param.numel()
            chunk = flat_grad[offset : offset + n].view_as(param).to(param.device, param.dtype)
            if param.grad is None:
                param.grad = chunk.clone()
            else:
                param.grad.copy_(chunk)
            offset += n
        if offset != flat_grad.numel():
            raise ValueError("flat_grad has unused elements")
