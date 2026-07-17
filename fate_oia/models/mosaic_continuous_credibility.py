"""Continuous, label-independent visual credibility for MOSAIC-TRUST v4.

This module deliberately does not accept reason/action targets.  It is a
visual measurement of whether a factor has stable, grounded evidence.  The
discrete certificate remains a deployment admission artifact, never a
training switch.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def absence_polarity(*, presence: float | torch.Tensor, observability: float | torch.Tensor) -> float | torch.Tensor:
    """Evidence for an observable absence, not positive presence evidence."""
    if isinstance(presence, torch.Tensor) or isinstance(observability, torch.Tensor):
        return (1.0 - torch.as_tensor(presence, dtype=torch.float32)).clamp(0.0, 1.0) * torch.as_tensor(
            observability, dtype=torch.float32
        ).clamp(0.0, 1.0)
    return float(max(0.0, min(1.0, 1.0 - float(presence))) * max(0.0, min(1.0, float(observability))))


def visual_credibility_from_measurements(
    *,
    content_score: torch.Tensor,
    prior_score: torch.Tensor,
    query_shuffle_score: torch.Tensor,
    image_shuffle_score: torch.Tensor,
    grounding_score: torch.Tensor,
    stability_score: torch.Tensor,
    n_eff: torch.Tensor,
    factor_role: str = "observable",
    reliable_negative: torch.Tensor | None = None,
    source_kind: str = "grounded",
) -> dict[str, torch.Tensor]:
    """Combine visual-only credibility measurements with explicit safety caps."""
    values = [content_score, prior_score, query_shuffle_score, image_shuffle_score, grounding_score, stability_score, n_eff]
    shape = content_score.shape
    if any(value.shape != shape for value in values):
        raise ValueError("visual credibility measurements must share one shape")
    if not all(torch.isfinite(value).all() for value in values):
        raise ValueError("visual credibility measurements must be finite")
    content = content_score.clamp(0.0, 1.0)
    prior = prior_score.clamp(0.0, 1.0)
    query = query_shuffle_score.clamp(0.0, 1.0)
    image = image_shuffle_score.clamp(0.0, 1.0)
    grounding = grounding_score.clamp(0.0, 1.0)
    stability = stability_score.clamp(0.0, 1.0)
    effective = n_eff.clamp_min(0.0)
    # Content and image intervention are primary; prior is only a supporting
    # measurement and cannot by itself certify a factor.
    credibility = (0.30 * content + 0.20 * query + 0.25 * image + 0.15 * grounding + 0.10 * stability)
    credibility = credibility * (effective / (effective + 8.0)).clamp(0.0, 1.0)
    if source_kind == "image_only":
        credibility = credibility.clamp_max(0.10)
    if factor_role in {"latent", "unsupported"}:
        credibility = credibility.clamp_max(0.25)
    if reliable_negative is not None:
        credibility = torch.where(reliable_negative.to(torch.bool), credibility, credibility.clamp_max(0.25))
    else:
        credibility = credibility.clamp_max(0.25)
    if source_kind == "unknown":
        credibility = torch.zeros_like(credibility)
    return {
        "cV": credibility.clamp(0.0, 1.0),
        "credibility_content": content,
        "credibility_prior": prior,
        "credibility_query_shuffle": query,
        "credibility_image_shuffle": image,
        "credibility_grounding": grounding,
        "credibility_stability": stability,
        "credibility_n_eff": effective,
    }


def update_credibility_ema(previous: torch.Tensor | None, current: torch.Tensor, decay: float = 0.90) -> torch.Tensor:
    if not 0.0 <= decay < 1.0:
        raise ValueError("credibility EMA decay must be in [0,1)")
    current = current.detach()
    if previous is None:
        return current.clone()
    if previous.shape != current.shape:
        raise ValueError("credibility EMA shape mismatch")
    return (decay * previous.detach() + (1.0 - decay) * current).detach()


class ContinuousVisualCredibility(nn.Module):
    """Batch-local cV estimate independent of labels and certificates."""

    def __init__(
        self,
        *,
        factor_count: int,
        dim: int,
        ema_decay: float = 0.90,
        factor_roles: list[str] | tuple[str, ...] | None = None,
        source_kinds: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        if factor_count <= 0 or dim <= 0:
            raise ValueError("factor_count and dim must be positive")
        self.factor_count = int(factor_count)
        self.dim = int(dim)
        self.ema_decay = float(ema_decay)
        factor_roles = tuple(factor_roles or ("observable",) * factor_count)
        source_kinds = tuple(source_kinds or ("grounded",) * factor_count)
        if len(factor_roles) != factor_count or len(source_kinds) != factor_count:
            raise ValueError("credibility factor metadata must match factor_count")
        caps = []
        for role, source_kind in zip(factor_roles, source_kinds):
            if source_kind == "unknown":
                caps.append(0.0)
            elif source_kind == "image_only":
                caps.append(0.10)
            elif role in {"latent", "unsupported"} or source_kind == "proxy":
                caps.append(0.25)
            else:
                caps.append(1.0)
        self.register_buffer("factor_credibility_cap", torch.tensor(caps, dtype=torch.float32), persistent=True)
        self.content_probe = nn.Linear(dim, 1)
        nn.init.zeros_(self.content_probe.bias)
        self.register_buffer("ema_cV", torch.zeros(factor_count), persistent=True)
        self.register_buffer("ema_initialized", torch.tensor(False), persistent=True)

    def forward(
        self,
        factor_features: torch.Tensor,
        presence_prob: torch.Tensor,
        uncertainty: torch.Tensor,
        *,
        grounding_score: torch.Tensor | None = None,
        sample_support: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if factor_features.ndim != 3 or factor_features.shape[1:] != (self.factor_count, self.dim):
            raise ValueError("factor_features must be [B,F,D]")
        if presence_prob.shape != uncertainty.shape or presence_prob.shape != factor_features.shape[:2]:
            raise ValueError("credibility probabilities must be [B,F]")
        content = torch.sigmoid(self.content_probe(factor_features).squeeze(-1))
        uncertainty = uncertainty.clamp(0.0, 1.0)
        confidence = (1.0 - uncertainty) * (0.5 + 0.5 * presence_prob.detach().clamp(0.0, 1.0))
        grounding = torch.ones_like(content) if grounding_score is None else grounding_score.to(content).clamp(0.0, 1.0)
        n_eff = torch.ones_like(content) if sample_support is None else sample_support.to(content).clamp_min(0.0)
        # These are image-only interventions: changing factor identity and
        # changing the image in the batch must alter the measured credibility.
        # They are diagnostic scores, not reason-label supervision.
        prior = presence_prob.detach().clamp(0.0, 1.0)
        query_shuffle = (1.0 - (content - content.roll(shifts=1, dims=1)).abs()).clamp(0.0, 1.0)
        image_shuffle = (1.0 - (content - content.roll(shifts=1, dims=0)).abs()).clamp(0.0, 1.0)
        stability = (1.0 - uncertainty).clamp(0.0, 1.0)
        support = (n_eff / (n_eff + 8.0)).clamp(0.0, 1.0)
        current = (
            0.25 * content
            + 0.05 * prior
            + 0.20 * query_shuffle
            + 0.20 * image_shuffle
            + 0.15 * grounding
            + 0.15 * stability
        ).clamp(0.0, 1.0)
        current = current * self.factor_credibility_cap.view(1, -1).to(current)
        # The EMA is a detached diagnostic prior; it never becomes a hard gate.
        batch_mean = current.detach().mean(0)
        if self.training:
            updated = update_credibility_ema(
                self.ema_cV if bool(self.ema_initialized) else None, batch_mean, self.ema_decay
            )
            self.ema_cV.copy_(updated)
            self.ema_initialized.fill_(True)
        return {
            "cV": current,
            "cV_ema": self.ema_cV.view(1, -1).expand_as(current),
            "cV_content": content,
            "cV_confidence": confidence,
            "cV_grounding": grounding,
            "cV_sample_support": support,
            "cV_n_eff": n_eff,
            "cV_prior": prior,
            "cV_query_shuffle_score": query_shuffle,
            "cV_image_shuffle_score": image_shuffle,
            "cV_stability": stability,
        }
