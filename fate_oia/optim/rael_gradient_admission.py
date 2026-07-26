"""P13 gradient admission for the two RAEL shared evidence boundaries.

This module deliberately has no optimizer, parameter iteration, or backward
call.  It obtains boundary gradients with ``autograd.grad`` and installs
one-shot tensor hooks only around the caller's single final ``backward``.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
import copy
import inspect
import math
import threading
from typing import Mapping

import torch
from torch import Tensor, nn


EVIDENCE_SLOT_COUNT = 20
REASON_TOKEN_COUNT = 21
REASON_BUDGET = 0.25
GROUNDING_BUDGET = 0.15
COUNTERFACTUAL_BUDGET = 0.05
_HAS_BATCHED_VJP = "is_grads_batched" in inspect.signature(torch.autograd.grad).parameters


def _validate_boundary(boundary: Tensor, *, slots: int, name: str) -> None:
    if not isinstance(boundary, Tensor) or boundary.ndim != 3 or boundary.shape[1] != slots:
        raise ValueError(f"{name} must be [B,{slots},D]")
    if not torch.is_floating_point(boundary):
        raise TypeError(f"{name} must be floating point")


def _validate_gradient(gradient: Tensor | None, action: Tensor, *, name: str) -> None:
    if gradient is None:
        return
    if not isinstance(gradient, Tensor) or gradient.shape != action.shape:
        raise ValueError(f"{name} must match action gradient shape")
    if gradient.device != action.device or not torch.is_floating_point(gradient):
        raise ValueError(f"{name} must be floating point on the action gradient device")


def per_sample_boundary_grads(loss: Tensor, boundaries: tuple[Tensor, ...]) -> tuple[Tensor | None, ...]:
    """Extract exact diagonal per-sample VJPs for all boundaries in one call."""

    if not isinstance(loss, Tensor):
        raise TypeError("loss must be a Tensor")
    if not boundaries:
        raise ValueError("at least one boundary is required")
    for boundary in boundaries:
        if not isinstance(boundary, Tensor) or boundary.ndim != 3:
            raise ValueError("boundary must be [B,K,D]")
    active_indices = [index for index, boundary in enumerate(boundaries) if boundary.requires_grad]
    if not loss.requires_grad or not active_indices:
        return tuple(None for _ in boundaries)
    active_boundaries = tuple(boundaries[index] for index in active_indices)
    if loss.ndim == 0:
        gradients = torch.autograd.grad(
            loss,
            active_boundaries,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
        result: list[Tensor | None] = [None] * len(boundaries)
        for index, gradient in zip(active_indices, gradients):
            result[index] = None if gradient is None else gradient.detach()
        return tuple(result)
    if loss.ndim != 1 or any(loss.shape[0] != boundary.shape[0] for boundary in boundaries):
        raise ValueError("loss must be scalar or [B]")
    if not _HAS_BATCHED_VJP:
        raise RuntimeError("P13 requires torch.autograd.grad(is_grads_batched=True); slow per-sample fallback is forbidden")
    batch = loss.shape[0]
    grad_outputs = torch.eye(batch, device=loss.device, dtype=loss.dtype)
    try:
        gradients = torch.autograd.grad(
            loss,
            active_boundaries,
            grad_outputs=grad_outputs,
            is_grads_batched=True,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )
    except TypeError as error:
        raise RuntimeError("P13 requires torch.autograd.grad(is_grads_batched=True); slow per-sample fallback is forbidden") from error
    diagonal = torch.arange(batch, device=loss.device)
    result = [None] * len(boundaries)
    for index, gradient in zip(active_indices, gradients):
        result[index] = None if gradient is None else gradient[diagonal, diagonal].detach()
    return tuple(result)


def per_sample_boundary_grad(loss: Tensor, boundary: Tensor) -> Tensor | None:
    """Return exact boundary gradients without touching ``boundary.grad``.

    A rank-one loss is interpreted as one already-weighted value per sample.
    Each scalar is differentiated separately, so the returned gradient keeps
    only the matching sample row.  A scalar loss is supported as its exact
    aggregate gradient; it remains a documented aggregate, not a fabricated
    per-sample decomposition.
    """

    return per_sample_boundary_grads(loss, (boundary,))[0]


def project_against_action_ema(gradient: Tensor, action_ema: Tensor, *, eps: float = 1.0e-8) -> Tensor:
    """Remove only an auxiliary component anti-aligned with the action EMA."""

    if gradient.ndim != 3 or action_ema.ndim != 2 or gradient.shape[1:] != action_ema.shape:
        raise ValueError("gradient must be [B,K,D] and action_ema must be [K,D]")
    if gradient.device != action_ema.device:
        raise ValueError("gradient and action_ema must share a device")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    g = gradient.float()
    anchor = action_ema.float().unsqueeze(0).expand_as(g)
    # This is exactly ``g - min(0, <g,m> / (||m||^2 + eps)) * m``.
    # Per-slot rescaling keeps both numerator and denominator finite for
    # very large EMA values without changing the epsilon-bearing formula.
    scale = anchor.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
    scaled_anchor = anchor / scale
    scaled_dot = (g * scaled_anchor).sum(dim=-1, keepdim=True)
    scaled_denom = scaled_anchor.square().sum(dim=-1, keepdim=True) + float(eps) / scale.square()
    negative_component = torch.minimum(torch.zeros_like(scaled_dot), scaled_dot / scaled_denom)
    return g - negative_component * scaled_anchor


def cap_auxiliary_gradient(
    projected_gradient: Tensor,
    action_ema: Tensor,
    *,
    ratio: float,
    eps: float = 1.0e-8,
) -> Tensor:
    """Cap each sample/slot auxiliary vector relative to ``||action_ema||``."""

    if projected_gradient.ndim != 3 or action_ema.ndim != 2 or projected_gradient.shape[1:] != action_ema.shape:
        raise ValueError("projected_gradient must be [B,K,D] and action_ema must be [K,D]")
    if projected_gradient.device != action_ema.device:
        raise ValueError("projected_gradient and action_ema must share a device")
    if ratio < 0.0 or eps <= 0.0:
        raise ValueError("ratio must be nonnegative and eps must be positive")
    gradient = projected_gradient.float()
    cap = float(ratio) * torch.linalg.vector_norm(action_ema.float(), dim=-1, keepdim=True).unsqueeze(0)
    norm = torch.linalg.vector_norm(gradient, dim=-1, keepdim=True)
    scale = torch.minimum(torch.ones_like(norm), cap / (norm + float(eps)))
    return gradient * scale


@dataclass(frozen=True)
class AdmissionCoreResult:
    projected_reason: Tensor
    projected_grounding: Tensor
    projected_counterfactual: Tensor
    admitted_reason: Tensor
    admitted_grounding: Tensor
    admitted_counterfactual: Tensor
    admitted: Tensor
    diagnostics: Mapping[str, Tensor]


def admission_core(
    action_gradient: Tensor,
    reason_gradient: Tensor,
    grounding_gradient: Tensor,
    counterfactual_gradient: Tensor,
    action_ema: Tensor,
    *,
    eps: float = 1.0e-8,
) -> AdmissionCoreResult:
    """Pure, compile-friendly P13 admission math; it holds no mutable state."""

    for name, gradient in (
        ("reason_gradient", reason_gradient),
        ("grounding_gradient", grounding_gradient),
        ("counterfactual_gradient", counterfactual_gradient),
    ):
        _validate_gradient(gradient, action_gradient, name=name)
    if action_gradient.ndim != 3 or action_ema.ndim != 2 or action_gradient.shape[1:] != action_ema.shape:
        raise ValueError("action_gradient must be [B,K,D] and action_ema must be [K,D]")
    projected_reason = project_against_action_ema(reason_gradient, action_ema, eps=eps)
    projected_grounding = project_against_action_ema(grounding_gradient, action_ema, eps=eps)
    projected_counterfactual = project_against_action_ema(counterfactual_gradient, action_ema, eps=eps)
    admitted_reason = cap_auxiliary_gradient(projected_reason, action_ema, ratio=REASON_BUDGET, eps=eps)
    admitted_grounding = cap_auxiliary_gradient(projected_grounding, action_ema, ratio=GROUNDING_BUDGET, eps=eps)
    admitted_counterfactual = cap_auxiliary_gradient(projected_counterfactual, action_ema, ratio=COUNTERFACTUAL_BUDGET, eps=eps)
    admitted = action_gradient.float() + admitted_reason + admitted_grounding + admitted_counterfactual
    diagnostics = {
        "action_ema_norm_mean": torch.linalg.vector_norm(action_ema.float(), dim=-1).mean().detach(),
        "reason_admitted_norm_mean": torch.linalg.vector_norm(admitted_reason, dim=-1).mean().detach(),
        "grounding_admitted_norm_mean": torch.linalg.vector_norm(admitted_grounding, dim=-1).mean().detach(),
        "counterfactual_admitted_norm_mean": torch.linalg.vector_norm(admitted_counterfactual, dim=-1).mean().detach(),
    }
    return AdmissionCoreResult(
        projected_reason=projected_reason.to(dtype=action_gradient.dtype),
        projected_grounding=projected_grounding.to(dtype=action_gradient.dtype),
        projected_counterfactual=projected_counterfactual.to(dtype=action_gradient.dtype),
        admitted_reason=admitted_reason.to(dtype=action_gradient.dtype),
        admitted_grounding=admitted_grounding.to(dtype=action_gradient.dtype),
        admitted_counterfactual=admitted_counterfactual.to(dtype=action_gradient.dtype),
        admitted=admitted.to(dtype=action_gradient.dtype),
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class BoundaryAdmission:
    admitted: Tensor | None
    action_gradient: Tensor | None
    reason_gradient: Tensor | None
    grounding_gradient: Tensor | None
    counterfactual_gradient: Tensor | None
    active: bool
    inactive_reason: str | None
    diagnostics: Mapping[str, Tensor]


@dataclass(frozen=True)
class DualBoundaryAdmission:
    evidence: BoundaryAdmission
    semantic: BoundaryAdmission


class SharedBoundaryGradientHooks(AbstractContextManager["SharedBoundaryGradientHooks"]):
    """One-shot replacement hooks for exactly the two shared RAEL boundaries."""

    def __init__(
        self,
        *,
        evidence_slots: Tensor,
        semantic_reason_tokens: Tensor,
        admission: DualBoundaryAdmission,
        backward_scale: float = 1.0,
    ) -> None:
        _validate_boundary(evidence_slots, slots=EVIDENCE_SLOT_COUNT, name="evidence_slots")
        _validate_boundary(semantic_reason_tokens, slots=REASON_TOKEN_COUNT, name="semantic_reason_tokens")
        self._entries = (
            ("evidence_slots", evidence_slots, admission.evidence.admitted if admission.evidence.active else None),
            ("semantic_reason_tokens", semantic_reason_tokens, admission.semantic.admitted if admission.semantic.active else None),
        )
        if isinstance(backward_scale, bool) or not math.isfinite(float(backward_scale)) or float(backward_scale) <= 0.0:
            raise ValueError("backward_scale must be finite and > 0")
        self._backward_scale = float(backward_scale)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self.active = False
        self._entered = False

    @staticmethod
    def _replacement_hook(
        name: str,
        replacement: Tensor,
        handle_ref: dict[str, torch.utils.hooks.RemovableHandle],
        backward_scale: float,
    ):
        replacement = replacement.detach() * float(backward_scale)

        def hook(incoming: Tensor) -> Tensor:
            if incoming.shape != replacement.shape or incoming.device != replacement.device:
                raise RuntimeError(f"P13 {name} hook received an incompatible gradient")
            # This context protects exactly one final backward.  Remove before
            # returning so a retained graph cannot receive the admission twice.
            handle = handle_ref.get("handle")
            if handle is not None:
                handle.remove()
            return replacement.to(dtype=incoming.dtype)

        return hook

    def __enter__(self) -> "SharedBoundaryGradientHooks":
        if self._entered:
            raise RuntimeError("P13 boundary hook context cannot be entered twice")
        self._entered = True
        try:
            for name, boundary, replacement in self._entries:
                if replacement is None:
                    continue
                handle_ref: dict[str, torch.utils.hooks.RemovableHandle] = {}
                handle = boundary.register_hook(self._replacement_hook(name, replacement, handle_ref, self._backward_scale))
                handle_ref["handle"] = handle
                self._handles.append(handle)
            self.active = True
            return self
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self.active = False

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False


class RAELGradientAdmission(nn.Module):
    """Checkpointable P13 EMA state and loss-to-boundary admission adapter."""

    def __init__(
        self,
        *,
        ema_decay: float = 0.95,
        eps: float = 1.0e-8,
        state_dtype: torch.dtype = torch.float32,
        state_device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 <= ema_decay < 1.0 or eps <= 0.0:
            raise ValueError("ema_decay must be in [0,1) and eps must be positive")
        self.ema_decay = float(ema_decay)
        self.eps = float(eps)
        if not isinstance(state_dtype, torch.dtype) or not torch.empty((), dtype=state_dtype).is_floating_point():
            raise TypeError("state_dtype must be a floating point torch.dtype")
        initial_device = torch.device("cpu") if state_device is None else torch.device(state_device)
        # This non-persistent buffer is the only destination policy for
        # uninitialized EMA state.  Unlike a plain attribute it follows
        # ``Module.to`` exactly, while never becoming checkpoint content.
        self.register_buffer("_state_anchor", torch.empty(0, dtype=state_dtype, device=initial_device), persistent=False)
        self.register_buffer("evidence_action_ema", torch.empty(0, dtype=state_dtype, device=initial_device), persistent=True)
        self.register_buffer("semantic_action_ema", torch.empty(0, dtype=state_dtype, device=initial_device), persistent=True)
        self.register_buffer("evidence_ema_updates", torch.zeros((), dtype=torch.long, device=initial_device), persistent=True)
        self.register_buffer("semantic_ema_updates", torch.zeros((), dtype=torch.long, device=initial_device), persistent=True)
        self._state_lock = threading.RLock()

    @property
    def state_dtype(self) -> torch.dtype:
        """Compatibility view backed by the movable state anchor."""

        return self._state_anchor.dtype

    @property
    def state_device(self) -> torch.device:
        """Compatibility view backed by the movable state anchor."""

        return self._state_anchor.device

    def __getstate__(self):
        # ``threading.RLock`` is intentionally not pickleable.  The lock is
        # process-local synchronization, so serialize only model state.
        with self._state_lock:
            state = self.__dict__.copy()
            state.pop("_state_lock", None)
            return state

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)
        self._state_lock = threading.RLock()

    def state_dict(self, *args, **kwargs):
        """Return a lock-consistent deep tensor snapshot.

        ``nn.Module.state_dict`` normally returns detached aliases of buffer
        storage.  EMA updates are mutable, so aliases could expose a mixed or
        later-mutated checkpoint.  Cloning while holding the same lock used by
        updates gives callers one atomic state step and retains standard
        destination/metadata semantics.
        """

        with self._state_lock:
            snapshot = super().state_dict(*args, **kwargs)
            for key, value in tuple(snapshot.items()):
                if isinstance(value, Tensor):
                    snapshot[key] = value.detach().clone(memory_format=torch.preserve_format)
            if hasattr(snapshot, "_metadata"):
                snapshot._metadata = copy.deepcopy(snapshot._metadata)
            return snapshot

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load EMA state atomically with the same policy as snapshotting."""

        with self._state_lock:
            return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        with self._state_lock:
            local_state = state_dict.copy()
            if hasattr(state_dict, "_metadata"):
                local_state._metadata = copy.deepcopy(state_dict._metadata)
            for name in ("evidence_action_ema", "semantic_action_ema"):
                incoming = local_state.get(prefix + name)
                if incoming is None:
                    continue
                existing = self._buffers[name]
                target_dtype = existing.dtype if existing.numel() else self._state_anchor.dtype
                target_device = existing.device if existing.numel() else self._state_anchor.device
                self._buffers[name] = torch.empty(incoming.shape, device=target_device, dtype=target_dtype)
                local_state[prefix + name] = incoming.detach().to(device=target_device, dtype=target_dtype)
            for name, ema_name in (("evidence_ema_updates", "evidence_action_ema"), ("semantic_ema_updates", "semantic_action_ema")):
                incoming = local_state.get(prefix + name)
                if incoming is None:
                    continue
                existing = self._buffers[name]
                target_device = self._buffers[ema_name].device
                self._buffers[name] = torch.empty(incoming.shape, device=target_device, dtype=existing.dtype)
                local_state[prefix + name] = incoming.detach().to(device=target_device, dtype=existing.dtype)
            super()._load_from_state_dict(local_state, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    @staticmethod
    def _assert_single_process() -> None:
        distributed = torch.distributed
        if distributed.is_available() and distributed.is_initialized() and distributed.get_world_size() > 1:
            raise RuntimeError("P13 is single-process only; multi-rank DDP requires an explicit EMA synchronization strategy")

    def _update_action_ema(self, *, name: str, action_gradient: Tensor) -> Tensor:
        if action_gradient is None:
            raise ValueError("action_gradient cannot be None when updating EMA")
        ema_name = f"{name}_action_ema"
        counter_name = f"{name}_ema_updates"
        previous = self._buffers[ema_name]
        target_shape = action_gradient.shape[1:]
        if previous.numel() == 0:
            previous = torch.zeros(target_shape, device=self._state_anchor.device, dtype=self._state_anchor.dtype)
        elif tuple(previous.shape) != tuple(target_shape):
            raise ValueError(f"P13 {name} boundary shape changed after EMA initialization")
        if previous.device != action_gradient.device:
            raise ValueError("P13 state_device must match the shared boundary device")
        updated = self.ema_decay * previous.float() + (1.0 - self.ema_decay) * action_gradient.detach().float().mean(dim=0)
        updated = updated.to(dtype=previous.dtype, device=previous.device)
        self._buffers[ema_name] = updated.detach()
        counter = self._buffers[counter_name]
        if counter.device != updated.device:
            counter = counter.to(device=updated.device)
        self._buffers[counter_name] = counter + 1
        return updated.detach()

    def _current_action_ema(self, *, name: str, action_gradient: Tensor) -> Tensor:
        """Read the boundary EMA without mutating it for an inactive action loss."""

        ema_name = f"{name}_action_ema"
        previous = self._buffers[ema_name]
        target_shape = action_gradient.shape[1:]
        if previous.numel() == 0:
            if self._state_anchor.device != action_gradient.device:
                raise ValueError("P13 state anchor must match the shared boundary device")
            return torch.zeros(target_shape, device=action_gradient.device, dtype=self._state_anchor.dtype)
        if tuple(previous.shape) != tuple(target_shape):
            raise ValueError(f"P13 {name} boundary shape changed after EMA initialization")
        if previous.device != action_gradient.device:
            raise ValueError("P13 state_device must match the shared boundary device")
        return previous.detach().to(dtype=torch.float32)

    def _admit_one_boundary(
        self,
        *,
        name: str,
        action_gradient: Tensor | None,
        reason_gradient: Tensor | None,
        grounding_gradient: Tensor | None,
        counterfactual_gradient: Tensor | None,
        slots: int,
        update_action_ema: bool = True,
        loss_activity: Mapping[str, bool] | None = None,
    ) -> BoundaryAdmission:
        if action_gradient is None:
            return BoundaryAdmission(
                admitted=None,
                action_gradient=None,
                reason_gradient=reason_gradient,
                grounding_gradient=grounding_gradient,
                counterfactual_gradient=counterfactual_gradient,
                active=False,
                inactive_reason="action_gradient_none",
                diagnostics={"active": torch.zeros((), dtype=torch.bool)},
            )
        _validate_boundary(action_gradient, slots=slots, name=f"{name}_action_gradient")
        for auxiliary_name, gradient in (
            ("reason", reason_gradient),
            ("grounding", grounding_gradient),
            ("counterfactual", counterfactual_gradient),
        ):
            _validate_gradient(gradient, action_gradient, name=f"{name}_{auxiliary_name}_gradient")
        zeros = torch.zeros_like(action_gradient)
        action_ema = (
            self._update_action_ema(name=name, action_gradient=action_gradient)
            if update_action_ema
            else self._current_action_ema(name=name, action_gradient=action_gradient)
        )
        core = admission_core(
            action_gradient,
            reason_gradient if reason_gradient is not None else zeros,
            grounding_gradient if grounding_gradient is not None else zeros,
            counterfactual_gradient if counterfactual_gradient is not None else zeros,
            action_ema,
            eps=self.eps,
        )
        diagnostics = dict(core.diagnostics)
        all_losses_none = loss_activity is not None and not any(loss_activity.values())
        diagnostics["active"] = torch.tensor(not all_losses_none, device=action_gradient.device, dtype=torch.bool)
        if loss_activity is not None:
            for loss_name, is_active in loss_activity.items():
                diagnostics[f"{loss_name}_loss_active"] = torch.tensor(
                    bool(is_active), device=action_gradient.device, dtype=torch.bool
                )
            diagnostics["all_losses_none"] = torch.tensor(
                all_losses_none, device=action_gradient.device, dtype=torch.bool
            )
        diagnostics["ema_update_count"] = self._buffers[f"{name}_ema_updates"].detach().clone()
        return BoundaryAdmission(
            admitted=core.admitted.detach(),
            action_gradient=action_gradient.detach(),
            reason_gradient=None if reason_gradient is None else reason_gradient.detach(),
            grounding_gradient=None if grounding_gradient is None else grounding_gradient.detach(),
            counterfactual_gradient=None if counterfactual_gradient is None else counterfactual_gradient.detach(),
            active=not all_losses_none,
            inactive_reason="all_losses_none" if all_losses_none else None,
            diagnostics={key: value.detach() for key, value in diagnostics.items()},
        )

    def admit_from_gradients(
        self,
        *,
        evidence_action: Tensor | None,
        evidence_reason: Tensor | None,
        evidence_grounding: Tensor | None,
        evidence_counterfactual: Tensor | None,
        semantic_action: Tensor | None,
        semantic_reason: Tensor | None,
        semantic_grounding: Tensor | None,
        semantic_counterfactual: Tensor | None,
    ) -> DualBoundaryAdmission:
        """Admit precomputed true gradients; no parameter ``.grad`` is read or written."""

        self._assert_single_process()
        return self._admit_dual(
            evidence_action=evidence_action,
            evidence_reason=evidence_reason,
            evidence_grounding=evidence_grounding,
            evidence_counterfactual=evidence_counterfactual,
            semantic_action=semantic_action,
            semantic_reason=semantic_reason,
            semantic_grounding=semantic_grounding,
            semantic_counterfactual=semantic_counterfactual,
        )

    def _admit_dual(
        self,
        *,
        evidence_action: Tensor | None,
        evidence_reason: Tensor | None,
        evidence_grounding: Tensor | None,
        evidence_counterfactual: Tensor | None,
        semantic_action: Tensor | None,
        semantic_reason: Tensor | None,
        semantic_grounding: Tensor | None,
        semantic_counterfactual: Tensor | None,
        update_action_ema: bool = True,
        loss_activity: Mapping[str, bool] | None = None,
    ) -> DualBoundaryAdmission:
        with self._state_lock:
            return DualBoundaryAdmission(
                evidence=self._admit_one_boundary(
                    name="evidence",
                    action_gradient=evidence_action,
                    reason_gradient=evidence_reason,
                    grounding_gradient=evidence_grounding,
                    counterfactual_gradient=evidence_counterfactual,
                    slots=EVIDENCE_SLOT_COUNT,
                    update_action_ema=update_action_ema,
                    loss_activity=loss_activity,
                ),
                semantic=self._admit_one_boundary(
                    name="semantic",
                    action_gradient=semantic_action,
                    reason_gradient=semantic_reason,
                    grounding_gradient=semantic_grounding,
                    counterfactual_gradient=semantic_counterfactual,
                    slots=REASON_TOKEN_COUNT,
                    update_action_ema=update_action_ema,
                    loss_activity=loss_activity,
                ),
            )

    def admit_from_losses(
        self,
        *,
        evidence_slots: Tensor,
        semantic_reason_tokens: Tensor,
        action_loss: Tensor | None,
        reason_loss: Tensor | None,
        grounding_loss: Tensor | None,
        counterfactual_loss: Tensor | None,
    ) -> DualBoundaryAdmission:
        """Extract weighted loss gradients with ``autograd.grad`` then admit them."""

        _validate_boundary(evidence_slots, slots=EVIDENCE_SLOT_COUNT, name="evidence_slots")
        _validate_boundary(semantic_reason_tokens, slots=REASON_TOKEN_COUNT, name="semantic_reason_tokens")
        losses = {
            "action": action_loss,
            "reason": reason_loss,
            "grounding": grounding_loss,
            "counterfactual": counterfactual_loss,
        }
        for name, loss in losses.items():
            if loss is not None and not isinstance(loss, Tensor):
                raise TypeError(f"{name}_loss must be a Tensor or None")
        loss_activity = {name: loss is not None for name, loss in losses.items()}

        self._assert_single_process()
        gradients_by_loss: dict[str, tuple[Tensor, Tensor]] = {}
        for name, loss in losses.items():
            if loss is None:
                gradients_by_loss[name] = (torch.zeros_like(evidence_slots), torch.zeros_like(semantic_reason_tokens))
                continue
            evidence_gradient, semantic_gradient = per_sample_boundary_grads(
                loss, (evidence_slots, semantic_reason_tokens)
            )
            gradients_by_loss[name] = (
                torch.zeros_like(evidence_slots) if evidence_gradient is None else evidence_gradient,
                torch.zeros_like(semantic_reason_tokens) if semantic_gradient is None else semantic_gradient,
            )
        return self._admit_dual(
            evidence_action=gradients_by_loss["action"][0],
            evidence_reason=gradients_by_loss["reason"][0],
            evidence_grounding=gradients_by_loss["grounding"][0],
            evidence_counterfactual=gradients_by_loss["counterfactual"][0],
            semantic_action=gradients_by_loss["action"][1],
            semantic_reason=gradients_by_loss["reason"][1],
            semantic_grounding=gradients_by_loss["grounding"][1],
            semantic_counterfactual=gradients_by_loss["counterfactual"][1],
            update_action_ema=loss_activity["action"],
            loss_activity=loss_activity,
        )

    def replace_shared_boundary_gradients(
        self,
        *,
        evidence_slots: Tensor,
        semantic_reason_tokens: Tensor,
        admission: DualBoundaryAdmission,
        backward_scale: float = 1.0,
    ) -> SharedBoundaryGradientHooks:
        """Return a context manager for the caller's single final backward."""

        return SharedBoundaryGradientHooks(
            evidence_slots=evidence_slots,
            semantic_reason_tokens=semantic_reason_tokens,
            admission=admission,
            backward_scale=backward_scale,
        )


__all__ = [
    "AdmissionCoreResult",
    "BoundaryAdmission",
    "COUNTERFACTUAL_BUDGET",
    "DualBoundaryAdmission",
    "EVIDENCE_SLOT_COUNT",
    "GROUNDING_BUDGET",
    "RAELGradientAdmission",
    "REASON_BUDGET",
    "REASON_TOKEN_COUNT",
    "SharedBoundaryGradientHooks",
    "admission_core",
    "cap_auxiliary_gradient",
    "per_sample_boundary_grad",
    "per_sample_boundary_grads",
    "project_against_action_ema",
]
