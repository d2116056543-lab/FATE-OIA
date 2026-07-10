from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


class MOSAICActionAnchoredGradient:
    def __init__(
        self,
        *,
        aux_shared_lambda_max: float = 0.25,
        action_anchor_kappa: float = 0.70,
        epsilon: float = 1e-12,
        numerical_tolerance: float = 1e-6,
    ) -> None:
        if not 0.0 <= aux_shared_lambda_max <= 1.0:
            raise ValueError("aux_shared_lambda_max must be in [0,1]")
        if not 0.0 <= action_anchor_kappa <= 1.0:
            raise ValueError("action_anchor_kappa must be in [0,1]")
        self.lambda_max = float(aux_shared_lambda_max)
        self.kappa = float(action_anchor_kappa)
        self.epsilon = float(epsilon)
        self.numerical_tolerance = float(numerical_tolerance)
        self.step_count = 0
        self.violation_count = 0
        self._pending: dict[str, Any] | None = None

    @staticmethod
    def _validate_partition(
        shared: Sequence[torch.nn.Parameter],
        action_only: Sequence[torch.nn.Parameter],
        explanation_only: Sequence[torch.nn.Parameter],
    ) -> None:
        partitions = [list(shared), list(action_only), list(explanation_only)]
        flattened = [parameter for partition in partitions for parameter in partition]
        if not partitions[0]:
            raise ValueError("action-anchored gradient requires shared parameters")
        if any(not isinstance(parameter, torch.nn.Parameter) or not parameter.requires_grad for parameter in flattened):
            raise ValueError("all action-anchor parameters must be trainable Parameters")
        identities = [id(parameter) for parameter in flattened]
        if len(set(identities)) != len(identities):
            raise ValueError("action-anchor parameter partitions must be unique and disjoint")

    @staticmethod
    def _replace_unused(
        gradients: Sequence[torch.Tensor | None],
        parameters: Sequence[torch.nn.Parameter],
    ) -> list[torch.Tensor]:
        return [
            torch.zeros_like(parameter) if gradient is None else gradient
            for gradient, parameter in zip(gradients, parameters)
        ]

    @staticmethod
    def _add_to_parameter(parameter: torch.nn.Parameter, gradient: torch.Tensor) -> None:
        detached = gradient.detach().to(device=parameter.device, dtype=parameter.dtype)
        if parameter.grad is None:
            parameter.grad = detached.clone()
        else:
            parameter.grad.add_(detached)

    @staticmethod
    def _sum_into(
        accumulated: list[torch.Tensor] | None,
        incoming: Sequence[torch.Tensor],
    ) -> list[torch.Tensor]:
        if accumulated is None:
            return [gradient.detach().clone() for gradient in incoming]
        for destination, gradient in zip(accumulated, incoming):
            destination.add_(gradient.detach())
        return accumulated

    def accumulate(
        self,
        action_loss: torch.Tensor,
        explanation_loss: torch.Tensor,
        shared_params: Sequence[torch.nn.Parameter],
        action_only_params: Sequence[torch.nn.Parameter],
        explanation_only_params: Sequence[torch.nn.Parameter],
        *,
        loss_scale: float = 1.0,
    ) -> None:
        shared = list(shared_params)
        action_only = list(action_only_params)
        explanation_only = list(explanation_only_params)
        self._validate_partition(shared, action_only, explanation_only)
        if type(loss_scale) not in {int, float} or float(loss_scale) <= 0:
            raise ValueError("loss_scale must be positive")
        partition_ids = tuple(
            tuple(id(parameter) for parameter in partition)
            for partition in (shared, action_only, explanation_only)
        )
        if self._pending is not None and self._pending["partition_ids"] != partition_ids:
            raise ValueError("action-anchor partitions changed inside an accumulation window")

        action_parameters = shared + action_only
        explanation_parameters = shared + explanation_only
        action_gradients = self._replace_unused(
            torch.autograd.grad(
                action_loss * float(loss_scale),
                action_parameters,
                retain_graph=True,
                allow_unused=True,
            ),
            action_parameters,
        )
        explanation_gradients = self._replace_unused(
            torch.autograd.grad(
                explanation_loss * float(loss_scale),
                explanation_parameters,
                allow_unused=True,
            ),
            explanation_parameters,
        )
        if self._pending is None:
            self._pending = {
                "partition_ids": partition_ids,
                "shared": shared,
                "action_only": action_only,
                "explanation_only": explanation_only,
                "action_shared": None,
                "explanation_shared": None,
                "action_specific": None,
                "explanation_specific": None,
                "microbatch_count": 0,
            }
        self._pending["action_shared"] = self._sum_into(
            self._pending["action_shared"], action_gradients[: len(shared)]
        )
        self._pending["explanation_shared"] = self._sum_into(
            self._pending["explanation_shared"], explanation_gradients[: len(shared)]
        )
        self._pending["action_specific"] = self._sum_into(
            self._pending["action_specific"], action_gradients[len(shared) :]
        )
        self._pending["explanation_specific"] = self._sum_into(
            self._pending["explanation_specific"], explanation_gradients[len(shared) :]
        )
        self._pending["microbatch_count"] += 1

    def finalize(self, *, step: int) -> dict[str, Any]:
        if type(step) is not int or step < 0:
            raise ValueError("step must be a non-negative integer")
        if self._pending is None:
            raise RuntimeError("cannot finalize an empty action-anchor accumulation window")
        pending = self._pending
        action_shared = pending["action_shared"]
        explanation_shared = pending["explanation_shared"]
        dot_action_aux = sum(
            (action_gradient.float() * explanation_gradient.float()).sum()
            for action_gradient, explanation_gradient in zip(action_shared, explanation_shared)
        )
        action_norm_squared = sum(gradient.float().square().sum() for gradient in action_shared)
        aux_norm_squared = sum(gradient.float().square().sum() for gradient in explanation_shared)
        conflict_limit = (1.0 - self.kappa) * action_norm_squared / (-dot_action_aux + self.epsilon)
        lambda_star = torch.where(
            dot_action_aux >= 0,
            dot_action_aux.new_tensor(self.lambda_max),
            conflict_limit.clamp(min=0.0, max=self.lambda_max),
        )
        combined_shared = [
            action_gradient + lambda_star.to(dtype=explanation_gradient.dtype) * explanation_gradient
            for action_gradient, explanation_gradient in zip(action_shared, explanation_shared)
        ]
        halfspace_lhs = sum(
            (combined.float() * action_gradient.float()).sum()
            for combined, action_gradient in zip(combined_shared, action_shared)
        )
        halfspace_rhs = self.kappa * action_norm_squared
        constraint_tensor = halfspace_lhs + self.numerical_tolerance >= halfspace_rhs
        diagnostics = torch.stack(
            (
                dot_action_aux,
                action_norm_squared.sqrt(),
                aux_norm_squared.sqrt(),
                lambda_star,
                halfspace_lhs,
                halfspace_rhs,
                constraint_tensor.to(dtype=halfspace_lhs.dtype),
            )
        ).detach().cpu().tolist()
        constraint_pass = bool(diagnostics[6])
        self.step_count += 1
        self.violation_count += int(not constraint_pass)

        for parameter, gradient in zip(pending["shared"], combined_shared):
            self._add_to_parameter(parameter, gradient)
        for parameter, gradient in zip(pending["action_only"], pending["action_specific"]):
            self._add_to_parameter(parameter, gradient)
        for parameter, gradient in zip(pending["explanation_only"], pending["explanation_specific"]):
            self._add_to_parameter(parameter, gradient)
        stats = {
            "step": step,
            "dot_action_aux": diagnostics[0],
            "action_grad_norm": diagnostics[1],
            "aux_grad_norm": diagnostics[2],
            "lambda_star": diagnostics[3],
            "halfspace_lhs": diagnostics[4],
            "halfspace_rhs": diagnostics[5],
            "constraint_pass": constraint_pass,
            "shared_param_count": len(pending["shared"]),
            "microbatch_count": pending["microbatch_count"],
        }
        self._pending = None
        return stats

    def backward(
        self,
        action_loss: torch.Tensor,
        explanation_loss: torch.Tensor,
        shared_params: Sequence[torch.nn.Parameter],
        action_only_params: Sequence[torch.nn.Parameter],
        explanation_only_params: Sequence[torch.nn.Parameter],
        *,
        step: int,
        loss_scale: float = 1.0,
    ) -> dict[str, Any]:
        self.accumulate(
            action_loss,
            explanation_loss,
            shared_params,
            action_only_params,
            explanation_only_params,
            loss_scale=loss_scale,
        )
        return self.finalize(step=step)

    @property
    def violation_rate(self) -> float:
        return self.violation_count / self.step_count if self.step_count else 0.0
