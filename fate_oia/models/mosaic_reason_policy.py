"""Bounded reason residual policy shared by the model and its audit tests."""

from __future__ import annotations

import math

import torch


def bounded_reason_residual(
    visual_logits: torch.Tensor,
    annotation_logits: torch.Tensor,
    *,
    init_alpha: float = 0.05,
    max_alpha: float = 0.25,
    alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add a bounded annotation residual without replacing visual logits.

    ``alpha`` is accepted as an already-bounded per-reason gate for the
    trainable mixer.  When omitted, the function uses the declared initial
    gate, which makes the standalone policy deterministic and zero-safe.
    """
    if visual_logits.shape != annotation_logits.shape or visual_logits.ndim < 1:
        raise ValueError("reason residual inputs must have matching non-empty shapes")
    if not 0.0 <= init_alpha <= max_alpha <= 1.0:
        raise ValueError("reason residual alpha bounds are invalid")
    channels = visual_logits.shape[-1]
    if alpha is None:
        gate = visual_logits.new_full((channels,), float(init_alpha))
    else:
        gate = torch.as_tensor(alpha, device=visual_logits.device, dtype=visual_logits.dtype)
        if gate.numel() not in {1, channels}:
            raise ValueError("reason residual alpha must be scalar or per-reason")
        gate = gate.reshape(-1).expand(channels).clamp(0.0, float(max_alpha))
    residual = gate.view(*([1] * (visual_logits.ndim - 1)), channels) * torch.tanh(
        annotation_logits - visual_logits
    )
    return visual_logits + residual, gate
