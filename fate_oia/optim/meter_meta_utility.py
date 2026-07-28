from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class METERMetaEvent:
    factor_ids: tuple[int, ...]
    action_only_loss: float
    action_reason_loss: float
    relative_utility: Tensor
    omega_before: Tensor
    omega_after: Tensor
    dino_calls: int
    action_grad_norm: float = 0.0
    reason_grad_norm: float = 0.0
    candidate_delta_norm: float = 0.0
    wall_time_sec: float = 0.0


class METERMetaUtility:
    """Train-audit virtual utility; it never mutates real parameters or optimizer state."""

    def __init__(self, factors: int = 21, virtual_lr: float = 1e-4, ema_old_weight: float = 0.9, ema_new_weight: float = 0.1, lower: float = 0.001, upper: float = 0.005) -> None:
        self.utility_ema = torch.zeros(factors)
        self.omega = torch.zeros(factors)
        self.cursor = 0
        self.virtual_lr = float(virtual_lr)
        self.ema_old_weight = float(ema_old_weight)
        self.ema_new_weight = float(ema_new_weight)
        self.lower = float(lower)
        self.upper = float(upper)

    @staticmethod
    def _clone_parameters(parameters: Mapping[str, Tensor]) -> dict[str, Tensor]:
        return {
            name: value.detach().clone().requires_grad_(True)
            for name, value in parameters.items()
        }

    @staticmethod
    def _gradient(
        loss_fn: Callable[[Mapping[str, Tensor]], Tensor],
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        loss = loss_fn(parameters)
        values = tuple(parameters.values())
        gradients = torch.autograd.grad(
            loss,
            values,
            create_graph=False,
            retain_graph=False,
            allow_unused=True,
        )
        return {
            name: (gradient if gradient is not None else torch.zeros_like(value))
            for (name, value), gradient in zip(parameters.items(), gradients)
        }

    def event(
        self,
        parameters: nn.Parameter | Mapping[str, Tensor],
        audit_loss: Callable[[Tensor], Tensor] | None = None,
        factor_ids: tuple[int, ...] = (),
        dino_calls: int = 0,
        *,
        action_loss_fn: Callable[[Mapping[str, Tensor]], Tensor] | None = None,
        reason_loss_fn: Callable[[Mapping[str, Tensor]], Tensor] | None = None,
        audit_action_loss_fn: Callable[[Mapping[str, Tensor]], Tensor] | None = None,
    ) -> METERMetaEvent:
        """Run one train-audit virtual utility event without mutating real state.

        The mapping API is the formal path: action and reason gradients are
        computed separately, both candidate adapter states are evaluated on
        the same encoded audit batch, and only ``omega``/EMA are updated.
        The scalar callback is retained only as a compatibility convenience
        for small unit tests; it deliberately has no synthetic reason gain.
        """
        started = time.perf_counter()
        action_grad_norm = 0.0
        reason_grad_norm = 0.0
        candidate_delta_norm = 0.0
        if isinstance(parameters, nn.Parameter):
            if audit_loss is None:
                raise ValueError("audit_loss is required for scalar compatibility mode")
            before = parameters.detach().clone().requires_grad_(True)
            gradient_action = torch.autograd.grad(
                audit_loss(before), before, create_graph=False, retain_graph=False, allow_unused=False
            )[0]
            candidate_action = before.detach() - self.virtual_lr * gradient_action
            action_loss = float(audit_loss(candidate_action).detach().cpu())
            candidate_reason = candidate_action
            reason_loss = float(audit_loss(candidate_reason).detach().cpu())
            raw_utility = (action_loss - reason_loss) / (abs(action_loss) + 1e-6)
            utility = torch.tensor(raw_utility)
        else:
            if action_loss_fn is None or reason_loss_fn is None:
                raise ValueError("Formal meta utility requires action_loss_fn and reason_loss_fn")
            before = self._clone_parameters(parameters)
            gradient_action = self._gradient(action_loss_fn, before)
            gradient_reason = self._gradient(reason_loss_fn, before)
            action_grad_norm = float(torch.sqrt(sum(value.float().square().sum() for value in gradient_action.values())).detach().cpu())
            reason_grad_norm = float(torch.sqrt(sum(value.float().square().sum() for value in gradient_reason.values())).detach().cpu())
            heldout = audit_action_loss_fn or action_loss_fn
            ids = tuple(int(x) for x in factor_ids)
            if not ids:
                raise ValueError("Formal meta utility requires at least one factor id")
            action_losses: list[float] = []
            reason_losses: list[float] = []
            utilities: list[Tensor] = []
            delta_norms: list[Tensor] = []
            for factor_id in ids:
                if factor_id < 0 or factor_id >= self.omega.numel():
                    raise IndexError(f"factor id out of range: {factor_id}")
                masked_action: dict[str, Tensor] = {}
                masked_reason: dict[str, Tensor] = {}
                for name, value in before.items():
                    if value.ndim == 0 or value.shape[0] != self.omega.numel():
                        raise ValueError(
                            f"Meta parameter {name!r} must have factor dimension first"
                        )
                    action_gradient = torch.zeros_like(gradient_action[name])
                    reason_gradient = torch.zeros_like(gradient_reason[name])
                    action_gradient[factor_id] = gradient_action[name][factor_id]
                    reason_gradient[factor_id] = gradient_reason[name][factor_id]
                    masked_action[name] = action_gradient
                    masked_reason[name] = reason_gradient
                candidate_action = {
                    name: value - self.virtual_lr * masked_action[name]
                    for name, value in before.items()
                }
                candidate_reason = {
                    name: value
                    - self.virtual_lr
                    * (masked_action[name] + masked_reason[name])
                    for name, value in before.items()
                }
                action_value = heldout(candidate_action)
                reason_value = heldout(candidate_reason)
                action_losses.append(float(action_value.detach().cpu()))
                reason_losses.append(float(reason_value.detach().cpu()))
                utilities.append(
                    (
                        (action_value - reason_value)
                        / (action_value.detach().abs() + 1e-6)
                    )
                    .detach()
                    .to(dtype=self.utility_ema.dtype, device="cpu")
                )
                delta_norms.append(
                    torch.sqrt(
                        sum(
                            (
                                candidate_reason[name]
                                - candidate_action[name]
                            )
                            .float()
                            .square()
                            .sum()
                            for name in candidate_reason
                        )
                    ).detach()
                )
            action_loss = float(sum(action_losses) / len(action_losses))
            reason_loss = float(sum(reason_losses) / len(reason_losses))
            utility = torch.stack(utilities)
            candidate_delta_norm = float(torch.stack(delta_norms).mean().cpu())

        ids = tuple(int(x) for x in factor_ids)
        omega_before = self.omega.clone()
        utility_by_factor = utility.reshape(-1)
        if utility_by_factor.numel() == 1 and len(ids) > 1:
            utility_by_factor = utility_by_factor.expand(len(ids))
        if utility_by_factor.numel() != len(ids):
            raise ValueError("Meta utility count does not match selected factors")
        for position, factor_id in enumerate(ids):
            if factor_id < 0 or factor_id >= self.omega.numel():
                raise IndexError(f"factor id out of range: {factor_id}")
            self.utility_ema[factor_id] = (
                self.ema_old_weight * self.utility_ema[factor_id]
                + self.ema_new_weight * utility_by_factor[position].detach().cpu()
            )
            denominator = max(self.upper - self.lower, 1e-6)
            self.omega[factor_id] = (
                (self.utility_ema[factor_id] - self.lower) / denominator
            ).clamp(0.0, 1.0)
        return METERMetaEvent(
            ids,
            action_loss,
            reason_loss,
            utility,
            omega_before,
            self.omega.clone(),
            int(dino_calls),
            action_grad_norm,
            reason_grad_norm,
            candidate_delta_norm,
            time.perf_counter() - started,
        )
