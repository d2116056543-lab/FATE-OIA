from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping

import torch
from torch import Tensor, nn


def shadow_adamw_update(
    value: Tensor,
    gradient: Tensor,
    *,
    exp_avg: Tensor,
    exp_avg_sq: Tensor,
    step: Tensor | int | float,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
) -> Tensor:
    """Return one AdamW candidate without mutating parameters or moments."""
    beta1, beta2 = (float(betas[0]), float(betas[1]))
    step_value = int(step.item()) if isinstance(step, Tensor) else int(step)
    next_step = step_value + 1
    next_exp_avg = beta1 * exp_avg + (1.0 - beta1) * gradient
    next_exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * gradient.square()
    bias_correction1 = 1.0 - beta1**next_step
    bias_correction2 = 1.0 - beta2**next_step
    denominator = next_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
    decayed = value * (1.0 - float(lr) * float(weight_decay))
    return decayed - (float(lr) / bias_correction1) * next_exp_avg / (
        denominator + float(eps)
    )


@dataclass(frozen=True)
class METERMetaEvent:
    factor_ids: tuple[int, ...]
    action_only_loss: float
    action_reason_loss: float
    relative_utility: Tensor
    omega_before: Tensor
    omega_after: Tensor
    dino_calls: int
    resolution_failure: Tensor
    observation_count: Tensor
    utility_ema_bias_corrected: Tensor
    shadow_update_used: bool
    lambda_m: float
    utility_samples: Tensor
    null_utility_samples: Tensor
    admission_lcb: Tensor
    null_q99: Tensor
    null_mad: Tensor
    admission_score: Tensor
    admitted: Tensor
    positive_streak: Tensor
    update_norm_normalized_utility: Tensor
    admission_mode: str
    actual_delta_norm: Tensor
    null_delta_norm: Tensor
    match_relative_error: Tensor
    control_observations: Tensor
    control_type: str
    invalid_observation: Tensor
    action_grad_norm: float = 0.0
    reason_grad_norm: float = 0.0
    candidate_delta_norm: float = 0.0
    wall_time_sec: float = 0.0


class METERMetaUtility:
    """Train-audit virtual utility; it never mutates real parameters or optimizer state."""

    def __init__(
        self,
        factors: int = 21,
        virtual_lr: float = 1e-4,
        ema_old_weight: float = 0.9,
        ema_new_weight: float = 0.1,
        lower: float = 0.001,
        upper: float = 0.005,
        *,
        admission_mode: str = "legacy_threshold",
        admission_min_consecutive: int = 2,
        admission_lcb_z: float = 1.645,
        admission_z_scale: float = 3.0,
        admission_eps: float = 1e-6,
        admission_min_observations: int = 8,
        admission_match_tolerance: float = 1e-4,
    ) -> None:
        self.utility_ema = torch.zeros(factors)
        self.omega = torch.zeros(factors)
        self.observation_count = torch.zeros(factors, dtype=torch.long)
        self.admission_score_ema = torch.zeros(factors)
        self.positive_streak = torch.zeros(factors, dtype=torch.long)
        self.cursor = 0
        self.virtual_lr = float(virtual_lr)
        self.ema_old_weight = float(ema_old_weight)
        self.ema_new_weight = float(ema_new_weight)
        self.lower = float(lower)
        self.upper = float(upper)
        if admission_mode not in {"legacy_threshold", "matched_null_lcb"}:
            raise ValueError(f"Unsupported meta admission mode: {admission_mode}")
        self.admission_mode = admission_mode
        self.admission_min_consecutive = int(admission_min_consecutive)
        self.admission_lcb_z = float(admission_lcb_z)
        self.admission_z_scale = float(admission_z_scale)
        self.admission_eps = float(admission_eps)
        self.admission_min_observations = int(admission_min_observations)
        self.admission_match_tolerance = float(admission_match_tolerance)

    @staticmethod
    def _loss_vector(value: Tensor) -> Tensor:
        if value.ndim == 0:
            return value.reshape(1)
        return value.reshape(-1)

    @staticmethod
    def _delta_norm(
        candidate: Mapping[str, Tensor],
        baseline: Mapping[str, Tensor],
    ) -> Tensor:
        return torch.sqrt(
            sum(
                (candidate[name] - baseline[name]).float().square().sum()
                for name in candidate
            )
        )

    @classmethod
    def _match_delta_norm(
        cls,
        candidate: Mapping[str, Tensor],
        baseline: Mapping[str, Tensor],
        target_norm: Tensor,
    ) -> dict[str, Tensor]:
        current_norm = cls._delta_norm(candidate, baseline)
        if bool(current_norm.eq(0)):
            return {name: value.clone() for name, value in baseline.items()}
        scale = (target_norm / current_norm).to(
            device=current_norm.device, dtype=current_norm.dtype
        )
        return {
            name: baseline[name] + (candidate[name] - baseline[name]) * scale
            for name in candidate
        }

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
        reason_loss_fn: Callable[[Mapping[str, Tensor], int], Tensor] | None = None,
        audit_action_loss_fn: Callable[[Mapping[str, Tensor]], Tensor] | None = None,
        shadow_optimizer_state: Mapping[str, Mapping[str, Any]] | None = None,
        lambda_m: float = 1.0,
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
        utility_samples_by_factor: list[Tensor] = []
        null_utility_samples_by_factor: list[Tensor] = []
        admission_lcb_values: list[Tensor] = []
        null_q99_values: list[Tensor] = []
        null_mad_values: list[Tensor] = []
        admission_score_values: list[Tensor] = []
        update_norm_values: list[Tensor] = []
        actual_delta_norm_values: list[Tensor] = []
        null_delta_norm_values: list[Tensor] = []
        match_relative_error_values: list[Tensor] = []
        control_observation_values: list[int] = []
        invalid_observation_values: list[bool] = []
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
            resolution_failure = torch.zeros(1, dtype=torch.bool)
            utility_samples_by_factor = [utility.reshape(1)]
            null_utility_samples_by_factor = [
                torch.full_like(utility.reshape(1), float("nan"))
            ]
            admission_lcb_values = [utility]
            null_q99_values = [utility.new_tensor(float("nan"))]
            null_mad_values = [utility.new_tensor(float("nan"))]
            admission_score_values = [utility.new_tensor(float("nan"))]
            update_norm_values = [utility.new_tensor(float("nan"))]
            actual_delta_norm_values = [utility.new_tensor(float("nan"))]
            null_delta_norm_values = [utility.new_tensor(float("nan"))]
            match_relative_error_values = [utility.new_tensor(float("nan"))]
            control_observation_values = [1]
            invalid_observation_values = [False]
        else:
            if action_loss_fn is None or reason_loss_fn is None:
                raise ValueError("Formal meta utility requires action_loss_fn and reason_loss_fn")
            before = self._clone_parameters(parameters)
            gradient_action = self._gradient(action_loss_fn, before)
            action_grad_norm = float(torch.sqrt(sum(value.float().square().sum() for value in gradient_action.values())).detach().cpu())
            heldout = audit_action_loss_fn or action_loss_fn
            ids = tuple(int(x) for x in factor_ids)
            if not ids:
                raise ValueError("Formal meta utility requires at least one factor id")
            action_losses: list[float] = []
            reason_losses: list[float] = []
            utilities: list[Tensor] = []
            delta_norms: list[Tensor] = []
            reason_grad_norms: list[Tensor] = []
            resolution_failures: list[bool] = []
            for factor_id in ids:
                if factor_id < 0 or factor_id >= self.omega.numel():
                    raise IndexError(f"factor id out of range: {factor_id}")
                gradient_reason = self._gradient(
                    lambda values: reason_loss_fn(values, factor_id),
                    before,
                )
                factor_reason_grad_norm = torch.sqrt(
                    sum(value.float().square().sum() for value in gradient_reason.values())
                ).detach()
                reason_grad_norms.append(factor_reason_grad_norm)
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
                candidate_action: dict[str, Tensor] = {}
                candidate_reason: dict[str, Tensor] = {}
                candidate_null: dict[str, Tensor] = {}
                for name, value in before.items():
                    if shadow_optimizer_state is None:
                        candidate_action[name] = (
                            value - self.virtual_lr * masked_action[name]
                        )
                        candidate_reason[name] = value - self.virtual_lr * (
                            masked_action[name] + float(lambda_m) * masked_reason[name]
                        )
                        candidate_null[name] = value - self.virtual_lr * (
                            masked_action[name] - float(lambda_m) * masked_reason[name]
                        )
                        continue
                    state = shadow_optimizer_state[name]
                    action_candidate = value.clone()
                    reason_candidate = value.clone()
                    null_candidate = value.clone()
                    common = {
                        "exp_avg": state["exp_avg"][factor_id],
                        "exp_avg_sq": state["exp_avg_sq"][factor_id],
                        "step": state["step"],
                        "lr": float(state["lr"]),
                        "betas": tuple(state["betas"]),
                        "eps": float(state["eps"]),
                        "weight_decay": float(state["weight_decay"]),
                    }
                    action_candidate[factor_id] = shadow_adamw_update(
                        value[factor_id],
                        masked_action[name][factor_id],
                        **common,
                    )
                    reason_candidate[factor_id] = shadow_adamw_update(
                        value[factor_id],
                        masked_action[name][factor_id]
                        + float(lambda_m) * masked_reason[name][factor_id],
                        **common,
                    )
                    null_candidate[factor_id] = shadow_adamw_update(
                        value[factor_id],
                        masked_action[name][factor_id]
                        - float(lambda_m) * masked_reason[name][factor_id],
                        **common,
                    )
                    candidate_action[name] = action_candidate
                    candidate_reason[name] = reason_candidate
                    candidate_null[name] = null_candidate
                candidate_delta = self._delta_norm(
                    candidate_reason, candidate_action
                ).detach()
                candidate_null = self._match_delta_norm(
                    candidate_null, candidate_action, candidate_delta
                )
                null_delta = self._delta_norm(
                    candidate_null, candidate_action
                ).detach()
                match_relative_error = (
                    (null_delta - candidate_delta).abs()
                    / candidate_delta.clamp_min(self.admission_eps)
                )
                # Held-out losses are observations only; no meta-gradient is
                # taken through these three candidate forwards.
                with torch.no_grad():
                    action_vector = self._loss_vector(heldout(candidate_action))
                    reason_vector = self._loss_vector(heldout(candidate_reason))
                    null_vector = self._loss_vector(heldout(candidate_null))
                if (
                    action_vector.shape != reason_vector.shape
                    or action_vector.shape != null_vector.shape
                ):
                    raise ValueError(
                        "Meta audit loss vectors must have identical shapes"
                    )
                action_value = action_vector.mean()
                reason_value = reason_vector.mean()
                action_losses.append(float(action_value.detach().cpu()))
                reason_losses.append(float(reason_value.detach().cpu()))
                loss_scale = torch.maximum(
                    torch.maximum(action_value.detach().abs(), reason_value.detach().abs()),
                    action_value.detach().new_ones(()),
                )
                resolution = torch.finfo(action_value.dtype).eps * loss_scale
                unresolved = bool(
                    factor_reason_grad_norm.gt(0)
                    and (
                        candidate_delta.eq(0)
                        or match_relative_error.gt(
                            self.admission_match_tolerance
                        )
                        or (action_vector.detach() - reason_vector.detach())
                        .abs()
                        .max()
                        .le(resolution)
                    )
                )
                resolution_failures.append(unresolved)
                actual_samples = (
                    (action_vector - reason_vector)
                    / (action_vector.detach().abs() + self.admission_eps)
                )
                null_samples = (
                    (action_vector - null_vector)
                    / (action_vector.detach().abs() + self.admission_eps)
                )
                actual_samples_cpu = actual_samples.detach().to(
                    dtype=self.utility_ema.dtype, device="cpu"
                )
                null_samples_cpu = null_samples.detach().to(
                    dtype=self.utility_ema.dtype, device="cpu"
                )
                utility_samples_by_factor.append(actual_samples_cpu)
                null_utility_samples_by_factor.append(null_samples_cpu)
                actual_delta_norm_values.append(candidate_delta.cpu())
                null_delta_norm_values.append(null_delta.cpu())
                match_relative_error_values.append(
                    match_relative_error.cpu()
                )
                control_observation_values.append(
                    int(actual_samples_cpu.numel())
                )
                invalid_observation_values.append(
                    unresolved
                    or (
                        self.admission_mode == "matched_null_lcb"
                        and actual_samples_cpu.numel()
                        < self.admission_min_observations
                    )
                )
                raw_mean = actual_samples_cpu.mean()
                utilities.append(
                    raw_mean.new_tensor(float("nan")) if unresolved else raw_mean
                )
                if (
                    actual_samples_cpu.numel()
                    >= self.admission_min_observations
                ):
                    standard_error = (
                        actual_samples_cpu.std(unbiased=True)
                        / math.sqrt(float(actual_samples_cpu.numel()))
                    )
                    actual_lcb = (
                        raw_mean - self.admission_lcb_z * standard_error
                    )
                else:
                    actual_lcb = raw_mean.new_tensor(float("-inf"))
                null_q99 = torch.quantile(null_samples_cpu, 0.99)
                null_median = torch.median(null_samples_cpu)
                null_mad = torch.median(
                    (null_samples_cpu - null_median).abs()
                )
                robust_scale = (
                    1.4826 * null_mad + self.admission_eps
                )
                z_score = (actual_lcb - null_q99) / robust_scale
                admission_score = (
                    (z_score / max(self.admission_z_scale, self.admission_eps))
                    .clamp(0.0, 1.0)
                    if bool(raw_mean.gt(0))
                    and bool(
                        actual_lcb.gt(
                            torch.maximum(
                                null_q99, null_q99.new_zeros(())
                            )
                        )
                    )
                    else raw_mean.new_zeros(())
                )
                admission_lcb_values.append(actual_lcb)
                null_q99_values.append(null_q99)
                null_mad_values.append(null_mad)
                admission_score_values.append(admission_score)
                update_norm_values.append(
                    raw_mean / candidate_delta.cpu().clamp_min(self.admission_eps)
                )
                delta_norms.append(candidate_delta)
            action_loss = float(sum(action_losses) / len(action_losses))
            reason_loss = float(sum(reason_losses) / len(reason_losses))
            utility = torch.stack(utilities)
            resolution_failure = torch.tensor(resolution_failures, dtype=torch.bool)
            candidate_delta_norm = float(torch.stack(delta_norms).mean().cpu())
            reason_grad_norm = float(torch.stack(reason_grad_norms).mean().cpu())

        ids = tuple(int(x) for x in factor_ids)
        omega_before = self.omega.clone()
        utility_by_factor = utility.reshape(-1)
        if utility_by_factor.numel() == 1 and len(ids) > 1:
            utility_by_factor = utility_by_factor.expand(len(ids))
        if utility_by_factor.numel() != len(ids):
            raise ValueError("Meta utility count does not match selected factors")
        admitted_values: list[bool] = []
        for position, factor_id in enumerate(ids):
            if factor_id < 0 or factor_id >= self.omega.numel():
                raise IndexError(f"factor id out of range: {factor_id}")
            utility_value = utility_by_factor[position].detach().cpu()
            if (
                not bool(torch.isfinite(utility_value))
                or invalid_observation_values[position]
            ):
                admitted_values.append(False)
                continue
            self.observation_count[factor_id] += 1
            self.utility_ema[factor_id] = (
                self.ema_old_weight * self.utility_ema[factor_id]
                + self.ema_new_weight * utility_value
            )
            count = int(self.observation_count[factor_id])
            correction = max(1.0 - self.ema_old_weight**count, 1e-12)
            bias_corrected = self.utility_ema[factor_id] / correction
            if self.admission_mode == "legacy_threshold":
                denominator = max(self.upper - self.lower, 1e-6)
                self.omega[factor_id] = (
                    (bias_corrected - self.lower) / denominator
                ).clamp(0.0, 1.0)
                admitted_values.append(bool(self.omega[factor_id].gt(0)))
                continue
            score = admission_score_values[position]
            positive = bool(score.gt(0)) and not bool(
                resolution_failure[position]
            )
            self.positive_streak[factor_id] = (
                self.positive_streak[factor_id] + 1
                if positive
                else 0
            )
            self.admission_score_ema[factor_id] = (
                self.ema_old_weight * self.admission_score_ema[factor_id]
                + self.ema_new_weight * score
            )
            admitted = bool(
                positive
                and self.positive_streak[factor_id]
                >= self.admission_min_consecutive
            )
            admitted_values.append(admitted)
            if admitted:
                self.omega[factor_id] = (
                    self.admission_score_ema[factor_id] / correction
                ).clamp(0.0, 1.0)
            else:
                self.omega[factor_id] = 0.0
        correction = 1.0 - torch.pow(
            torch.full_like(self.utility_ema, self.ema_old_weight),
            self.observation_count.to(dtype=self.utility_ema.dtype),
        )
        utility_ema_bias_corrected = torch.where(
            self.observation_count > 0,
            self.utility_ema / correction.clamp_min(1e-12),
            torch.zeros_like(self.utility_ema),
        )
        max_samples = max(value.numel() for value in utility_samples_by_factor)

        def stack_samples(values: list[Tensor]) -> Tensor:
            rows = []
            for value in values:
                flat = value.reshape(-1)
                if flat.numel() < max_samples:
                    flat = torch.cat(
                        (
                            flat,
                            torch.full(
                                (max_samples - flat.numel(),),
                                float("nan"),
                                dtype=flat.dtype,
                            ),
                        )
                    )
                rows.append(flat)
            return torch.stack(rows)

        utility_samples = stack_samples(utility_samples_by_factor)
        null_utility_samples = stack_samples(null_utility_samples_by_factor)
        if self.admission_mode == "legacy_threshold":
            admission_lcb = utility_by_factor.clone()
            null_q99 = torch.full_like(utility_by_factor, float("nan"))
            null_mad = torch.full_like(utility_by_factor, float("nan"))
            admission_score = self.omega[list(ids)].clone()
            update_norm_normalized_utility = torch.full_like(
                utility_by_factor, float("nan")
            )
        else:
            admission_lcb = torch.stack(admission_lcb_values)
            null_q99 = torch.stack(null_q99_values)
            null_mad = torch.stack(null_mad_values)
            admission_score = torch.stack(admission_score_values)
            update_norm_normalized_utility = torch.stack(update_norm_values)
        actual_delta_norm = torch.stack(actual_delta_norm_values)
        null_delta_norm = torch.stack(null_delta_norm_values)
        match_relative_error = torch.stack(match_relative_error_values)
        control_observations = torch.tensor(
            control_observation_values, dtype=torch.long
        )
        invalid_observation = torch.tensor(
            invalid_observation_values, dtype=torch.bool
        )
        return METERMetaEvent(
            ids,
            action_loss,
            reason_loss,
            utility,
            omega_before,
            self.omega.clone(),
            int(dino_calls),
            resolution_failure,
            self.observation_count.clone(),
            utility_ema_bias_corrected,
            shadow_optimizer_state is not None,
            float(lambda_m),
            utility_samples,
            null_utility_samples,
            admission_lcb,
            null_q99,
            null_mad,
            admission_score,
            torch.tensor(admitted_values, dtype=torch.bool),
            self.positive_streak.clone(),
            update_norm_normalized_utility,
            self.admission_mode,
            actual_delta_norm,
            null_delta_norm,
            match_relative_error,
            control_observations,
            "antipodal_sign_flip_equal_norm",
            invalid_observation,
            action_grad_norm,
            reason_grad_norm,
            candidate_delta_norm,
            time.perf_counter() - started,
        )
