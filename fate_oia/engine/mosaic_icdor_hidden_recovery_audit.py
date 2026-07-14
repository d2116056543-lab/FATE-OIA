"""Leakage-free audit metrics for synthetically hidden reason positives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch


MISSINGNESS_MODES = ("mcar", "mar", "mnar", "null")
SCHEMA_VERSION = "icdor_hidden_recovery_audit.v1"


def _ap(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    labels = labels.bool()
    positives = int(labels.sum())
    if positives == 0 or positives == labels.numel():
        return None
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = labels[order].float()
    return float((ranked.cumsum(0).div(torch.arange(1, ranked.numel() + 1, device=scores.device)) * ranked).sum() / positives)


def build_hidden_mask(
    observed_positive: torch.Tensor,
    *,
    mode: str,
    hide_fraction: float,
    propensity: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create audit-only missingness masks; hidden labels never enter a loss."""
    if mode not in MISSINGNESS_MODES or observed_positive.ndim != 2 or not 0.0 <= hide_fraction <= 1.0:
        raise ValueError("hidden recovery audit requires a supported mode, [B,R] labels, and fraction in [0,1]")
    positive = observed_positive > 0.5
    if mode == "null":
        return torch.zeros_like(positive)
    weights = torch.ones_like(observed_positive, dtype=torch.float32)
    if mode in {"mar", "mnar"}:
        if propensity is None or propensity.shape != observed_positive.shape:
            raise ValueError("MAR/MNAR hidden recovery audit requires aligned propensity")
        weights = propensity.float().clamp(0.0, 1.0)
        if mode == "mnar":
            weights = weights * positive.float() + (1.0 - weights) * (~positive).float()
    draw = torch.rand(observed_positive.shape, device=observed_positive.device, generator=generator)
    return positive & (draw < (hide_fraction * weights).clamp(max=1.0))


def audit_hidden_recovery(
    logits: torch.Tensor,
    observed_targets: torch.Tensor,
    *,
    propensity: torch.Tensor | None = None,
    hide_fraction: float = 0.25,
    generator: torch.Generator | None = None,
) -> dict[str, Any]:
    if logits.shape != observed_targets.shape or logits.ndim != 2:
        raise ValueError("hidden recovery audit requires matching [B,R] logits and observed targets")
    probability = logits.sigmoid().detach()
    modes: dict[str, Mapping[str, Any]] = {}
    for mode in MISSINGNESS_MODES:
        hidden = build_hidden_mask(observed_targets, mode=mode, hide_fraction=hide_fraction, propensity=propensity, generator=generator)
        scores = probability[hidden]
        labels = (observed_targets > 0.5)[hidden]
        modes[mode] = {"hidden_count": int(hidden.sum()), "recovery_mean": float(scores.mean()) if scores.numel() else None, "recovery_ap": _ap(scores, labels)}
    positives = observed_targets > 0.5
    modes["ceiling"] = {"hidden_count": int(positives.sum()), "recovery_mean": float(probability[positives].mean()) if bool(positives.any()) else None, "recovery_ap": _ap(probability.flatten(), positives.flatten())}
    return {"schema_version": SCHEMA_VERSION, "training_safe": True, "modes": modes}


collect_hidden_recovery_audit = audit_hidden_recovery
