from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MOSAICActionParetoAdmission(nn.Module):
    """Per-action dual variables preventing admitted routes from harming AP."""

    def __init__(self, *, action_count: int = 4, tolerance: float = 0.001, dual_lr: float = 0.05) -> None:
        super().__init__()
        if action_count != 4 or tolerance < 0.0 or dual_lr <= 0.0:
            raise ValueError("IC-DOR Pareto admission requires four actions, nonnegative tolerance, and positive dual LR")
        self.action_count = action_count
        self.tolerance = float(tolerance)
        self.dual_lr = float(dual_lr)
        self.register_buffer("dual_variables", torch.zeros(action_count), persistent=True)

    @torch.no_grad()
    def update_from_audit(self, visual_ap: torch.Tensor, routed_ap: torch.Tensor) -> dict[str, float]:
        if visual_ap.shape != (self.action_count,) or routed_ap.shape != (self.action_count,):
            raise ValueError("IC-DOR Pareto AP audit must be [4]")
        violation = visual_ap.detach() - routed_ap.detach() - self.tolerance
        self.dual_variables.add_(self.dual_lr * violation).clamp_(min=0.0)
        return {
            "pareto_violation_rate": float((violation >= 0.0).float().mean().cpu()),
            "pareto_violation_mean": float(violation.clamp_min(0.0).mean().cpu()),
            "pareto_dual_mean": float(self.dual_variables.mean().cpu()),
        }

    def route_penalty(
        self,
        visual_logits: torch.Tensor,
        routed_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Return a differentiable routed-vs-visual non-regression penalty."""
        if visual_logits.shape != routed_logits.shape or visual_logits.shape != targets.shape:
            raise ValueError("IC-DOR Pareto penalty requires matching [B,4] tensors")
        if visual_logits.ndim != 2 or visual_logits.shape[1] != self.action_count:
            raise ValueError("IC-DOR Pareto penalty requires four action logits")
        target = targets.to(dtype=routed_logits.dtype)
        visual_loss = F.binary_cross_entropy_with_logits(
            visual_logits.detach(), target, reduction="none"
        ).mean(dim=0)
        routed_loss = F.binary_cross_entropy_with_logits(
            routed_logits, target, reduction="none"
        ).mean(dim=0)
        violation = (routed_loss - visual_loss - self.tolerance).clamp_min(0.0)
        # Keep a primal non-regression cost even before audit duals activate.
        # The visual baseline is detached above, so gradients remain route-only.
        return ((1.0 + self.dual_variables.detach()) * violation).mean()
