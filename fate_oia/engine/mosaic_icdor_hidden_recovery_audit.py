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


def audit_hidden_recovery_scores(
    posterior_scores: torch.Tensor,
    zero_as_negative_scores: torch.Tensor,
    observed_after_hiding: torch.Tensor,
    hidden_positive_mask: torch.Tensor,
    *,
    mode: str,
    hide_fraction: float,
) -> dict[str, Any]:
    """Compare recovered hidden positives against eligible observed-zero rows.

    Hidden truths select the evaluation positives only. They are never passed
    back to the posterior model or a training loss.
    """
    shape = posterior_scores.shape
    if (
        mode not in MISSINGNESS_MODES
        or hide_fraction not in (0.10, 0.30, 0.50)
        or posterior_scores.ndim != 2
        or zero_as_negative_scores.shape != shape
        or observed_after_hiding.shape != shape
        or hidden_positive_mask.shape != shape
    ):
        raise ValueError("hidden recovery score audit requires aligned [B,R] tensors and a supported mode")
    hidden = hidden_positive_mask.bool()
    eligible_negative = (observed_after_hiding <= 0.5) & ~hidden
    per_label: list[dict[str, Any]] = []
    for reason_id in range(shape[1]):
        label_hidden = hidden[:, reason_id]
        label_negative = eligible_negative[:, reason_id]
        label_evaluation = label_hidden | label_negative
        label_hidden_count = int(label_hidden.sum())
        label_negative_count = int(label_negative.sum())
        if label_hidden_count == 0 or label_negative_count == 0:
            per_label.append({
                "reason_id": reason_id,
                "available": False,
                "hidden_positive_count": label_hidden_count,
                "eligible_negative_count": label_negative_count,
                "recovery_auprc": None,
                "zero_as_negative_auprc": None,
                "margin": None,
            })
            continue
        label_truth = label_hidden[label_evaluation]
        label_recovery = _ap(posterior_scores[:, reason_id][label_evaluation].detach(), label_truth)
        label_baseline = _ap(zero_as_negative_scores[:, reason_id][label_evaluation].detach(), label_truth)
        if label_recovery is None or label_baseline is None:
            raise ValueError("hidden recovery label audit unexpectedly produced an unavailable AP")
        per_label.append({
            "reason_id": reason_id,
            "available": True,
            "hidden_positive_count": label_hidden_count,
            "eligible_negative_count": label_negative_count,
            "recovery_auprc": label_recovery,
            "zero_as_negative_auprc": label_baseline,
            "margin": label_recovery - label_baseline,
        })
    evaluation = hidden | eligible_negative
    labels = hidden[evaluation]
    hidden_count = int(hidden.sum())
    negative_count = int(eligible_negative.sum())
    if hidden_count == 0 or negative_count == 0:
        return {
            "mode": mode,
            "hide_fraction": hide_fraction,
            "evaluation_only": True,
            "available": False,
            "hidden_positive_count": hidden_count,
            "eligible_negative_count": negative_count,
            "recovery_auprc": None,
            "zero_as_negative_auprc": None,
            "margin": None,
            "per_label": per_label,
            "training_safe": True,
        }
    recovery = _ap(posterior_scores[evaluation].detach(), labels)
    baseline = _ap(zero_as_negative_scores[evaluation].detach(), labels)
    if recovery is None or baseline is None:
        raise ValueError("hidden recovery score audit unexpectedly produced an unavailable AP")
    return {
        "mode": mode,
        "hide_fraction": hide_fraction,
        "evaluation_only": True,
        "available": True,
        "hidden_positive_count": hidden_count,
        "eligible_negative_count": negative_count,
        "recovery_auprc": recovery,
        "zero_as_negative_auprc": baseline,
        "margin": recovery - baseline,
        "per_label": per_label,
        "training_safe": True,
    }


def build_hidden_mask(
    observed_positive: torch.Tensor,
    *,
    mode: str,
    hide_fraction: float,
    propensity: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Create audit-only missingness masks; hidden labels never enter a loss."""
    if (
        mode not in MISSINGNESS_MODES
        or observed_positive.ndim != 2
        or hide_fraction not in (0.10, 0.30, 0.50)
    ):
        raise ValueError("hidden recovery audit requires a supported mode, [B,R] labels, and planned 10/30/50 fraction")
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
    hide_fraction: float = 0.10,
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
