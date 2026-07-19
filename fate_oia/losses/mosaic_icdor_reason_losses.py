from __future__ import annotations

import torch
from torch.nn import functional as F

from .mosaic_icdor_action_losses import asymmetric_loss


def _masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if values.shape != valid_mask.shape:
        raise ValueError("IC-DOR reason loss values and valid mask must match")
    valid = valid_mask.to(dtype=values.dtype)
    denominator = valid.sum()
    if not bool(denominator > 0):
        return values.sum() * 0.0
    return (values * valid).sum() / denominator


def _asymmetric_values(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """ASL terms before reduction so unobserved labels never become negatives."""
    probability = torch.sigmoid(logits)
    positive = targets * torch.nn.functional.logsigmoid(logits)
    negative_probability = (probability - 0.05).clamp_min(0.0)
    negative = (1.0 - targets) * negative_probability.pow(4.0) * torch.log1p(
        -negative_probability.clamp(max=1.0 - 1e-6)
    )
    return -(positive + negative)


def _per_label_balanced_asl(
    values: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Balance known positives and negatives independently for each reason."""
    if values.shape != targets.shape or values.shape != valid_mask.shape:
        raise ValueError("per-label IC-DOR ASL requires matching value, target, and valid tensors")
    label_losses: list[torch.Tensor] = []
    for reason_id in range(values.shape[1]):
        valid = valid_mask[:, reason_id]
        positive = valid & (targets[:, reason_id] > 0.5)
        negative = valid & ~positive
        class_terms: list[torch.Tensor] = []
        if bool(positive.any()):
            class_terms.append(values[positive, reason_id].mean())
        if bool(negative.any()):
            class_terms.append(values[negative, reason_id].mean())
        if class_terms:
            label_losses.append(torch.stack(class_terms).mean())
    return torch.stack(label_losses).mean() if label_losses else values.sum() * 0.0


def build_synthetic_hidden_positive_mask(
    observed_reason_targets: torch.Tensor,
    *,
    hide_fraction: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Hide a real subset of observed positives for PU recovery diagnostics."""
    if observed_reason_targets.ndim != 2 or observed_reason_targets.shape[1] != 21:
        raise ValueError("IC-DOR synthetic hiding requires observed reason targets [B,21]")
    if not 0.0 <= hide_fraction < 1.0:
        raise ValueError("IC-DOR synthetic hide fraction must be in [0,1)")
    positive = observed_reason_targets > 0.5
    draw = torch.rand(positive.shape, device=positive.device, generator=generator)
    hidden = positive & (draw < hide_fraction)
    # Preserve at least one observed positive per non-empty sample whenever possible.
    for row in range(hidden.shape[0]):
        indices = torch.nonzero(positive[row], as_tuple=False).flatten()
        if indices.numel() and hidden[row, indices].all():
            hidden[row, indices[0]] = False
    return hidden


def reason_observed_losses(
    reason_visual_observed_logits: torch.Tensor,
    reason_observed_logits: torch.Tensor,
    observed_reason_targets: torch.Tensor,
    *,
    observed_valid_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if reason_visual_observed_logits.shape != reason_observed_logits.shape or reason_visual_observed_logits.shape != observed_reason_targets.shape:
        raise ValueError("IC-DOR observed reason losses require matching [B,21] tensors")
    valid = torch.ones_like(observed_reason_targets, dtype=torch.bool) if observed_valid_mask is None else observed_valid_mask
    if valid.shape != observed_reason_targets.shape or valid.dtype != torch.bool:
        raise ValueError("IC-DOR observed reason valid mask must be bool [B,21]")
    targets = observed_reason_targets.to(dtype=reason_observed_logits.dtype)
    visual = _per_label_balanced_asl(_asymmetric_values(reason_visual_observed_logits, targets), targets, valid)
    observed = _per_label_balanced_asl(_asymmetric_values(reason_observed_logits, targets), targets, valid)
    return {
        "loss_reason_visual_observed_asl": visual,
        "loss_reason_observed_asl": observed,
        "loss_reason_observed_total": 0.50 * visual + observed,
    }


def latent_reason_core_loss(
    reason_logits_latent: torch.Tensor,
    observed_reason_targets: torch.Tensor,
    *,
    pu_enabled: bool,
    observed_valid_mask: torch.Tensor | None = None,
    weak_negative_weight: float = 0.15,
) -> torch.Tensor:
    """Train the latent semantic route even when PU recovery is unavailable.

    A zero BDD-OIA annotation remains a weak negative for this semantic core;
    the optional PU branch may subsequently recover missing positives per
    label, but a global failed hidden-label diagnostic must never silence this
    owner entirely.
    """
    if reason_logits_latent.shape != observed_reason_targets.shape or reason_logits_latent.shape[-1] != 21:
        raise ValueError("IC-DOR latent core loss requires matching [B,21] tensors")
    valid = torch.ones_like(observed_reason_targets, dtype=torch.bool) if observed_valid_mask is None else observed_valid_mask
    if valid.shape != observed_reason_targets.shape or valid.dtype != torch.bool:
        raise ValueError("IC-DOR latent core valid mask must be bool [B,21]")
    targets = observed_reason_targets.to(dtype=reason_logits_latent.dtype)
    positive = targets * F.softplus(-reason_logits_latent)
    negative = (1.0 - targets) * float(weak_negative_weight) * F.softplus(reason_logits_latent)
    # The result deliberately does not depend on ``pu_enabled``.  Keep the
    # argument in the public contract so artifacts can explicitly prove this.
    del pu_enabled
    return _per_label_balanced_asl(positive + negative, targets, valid)


def build_per_label_pu_gate(hidden_recovery_margin: torch.Tensor, *, minimum_margin: float = 0.0) -> torch.Tensor:
    """Admit PU recovery independently for each reason label."""
    if hidden_recovery_margin.ndim != 1 or hidden_recovery_margin.numel() != 21:
        raise ValueError("IC-DOR PU gate requires a [21] hidden-recovery margin")
    return torch.isfinite(hidden_recovery_margin) & (hidden_recovery_margin > float(minimum_margin))


def _posterior_rank_loss(logits: torch.Tensor, posterior: torch.Tensor, valid_mask: torch.Tensor, *, margin: float = 0.10) -> torch.Tensor:
    weights_positive = posterior.detach()
    weights_negative = 1.0 - posterior.detach()
    positive = torch.sigmoid(logits)
    losses: list[torch.Tensor] = []
    for reason_id in range(logits.shape[1]):
        keep = valid_mask[:, reason_id]
        p = positive[keep, reason_id]
        q_pos = weights_positive[keep, reason_id]
        q_neg = weights_negative[keep, reason_id]
        pair_weight = q_pos[:, None] * q_neg[None, :]
        if pair_weight.sum().item() > 0:
            losses.append((F.relu(margin - p[:, None] + p[None, :]) * pair_weight).sum() / pair_weight.sum())
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def selective_observation_losses(
    reason_logits_latent: torch.Tensor,
    observed_reason_targets: torch.Tensor,
    reason_observation_probability: torch.Tensor,
    posterior: torch.Tensor,
    reason_observation_logits: torch.Tensor | None = None,
    *,
    reason_propensity: torch.Tensor,
    factor_route_support: torch.Tensor,
    escape_weight: torch.Tensor,
    synthetic_hidden_positive_mask: torch.Tensor | None = None,
    observed_valid_mask: torch.Tensor | None = None,
    pu_gate: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    shapes = [observed_reason_targets, reason_observation_probability, posterior, reason_propensity, factor_route_support, escape_weight]
    if reason_logits_latent.ndim != 2 or reason_logits_latent.shape[-1] != 21 or any(value.shape != reason_logits_latent.shape for value in shapes):
        raise ValueError("IC-DOR selective observation losses require matching [B,21] tensors")
    if reason_observation_logits is not None and reason_observation_logits.shape != reason_logits_latent.shape:
        raise ValueError("IC-DOR observation logits must match latent reason logits [B,21]")
    valid = torch.ones_like(observed_reason_targets, dtype=torch.bool) if observed_valid_mask is None else observed_valid_mask
    if valid.shape != reason_logits_latent.shape or valid.dtype != torch.bool:
        raise ValueError("IC-DOR selective observation valid mask must be bool [B,21]")
    if pu_gate is None:
        pu_valid = valid
    else:
        if pu_gate.shape != (21,) or pu_gate.dtype != torch.bool:
            raise ValueError("IC-DOR PU gate must be bool [21]")
        pu_valid = valid & pu_gate.view(1, -1)
    target = observed_reason_targets.to(dtype=reason_logits_latent.dtype)
    observation_probability = reason_observation_probability.clamp(1e-6, 1.0 - 1e-6)
    # BCE on probabilities is unsafe under bf16 autocast. Prefer the model's
    # native observation logits; retain a float logit fallback for old callers.
    observation_logits = (
        reason_observation_logits
        if reason_observation_logits is not None
        else torch.logit(observation_probability.float())
    )
    # Kept for call-site compatibility only. Hidden labels are audit targets, never training loss targets.
    if synthetic_hidden_positive_mask is not None and (synthetic_hidden_positive_mask.shape != reason_logits_latent.shape or synthetic_hidden_positive_mask.dtype != torch.bool):
        raise ValueError("IC-DOR synthetic hidden-positive mask must be bool [B,21]")
    # Observation NLL trains the annotated branch, not hidden-positive recovery.
    # A failed PU audit may disable posterior recovery per label, but must never
    # erase supervision for labels that are actually observed in this batch.
    loss_nll = _masked_mean(F.binary_cross_entropy_with_logits(observation_logits, target, reduction="none"), valid)
    loss_posterior = _masked_mean(F.binary_cross_entropy_with_logits(reason_logits_latent, posterior.detach(), reduction="none"), pu_valid)
    loss_rank = _posterior_rank_loss(reason_logits_latent, posterior, pu_valid)
    loss_factor_consistency = _masked_mean((torch.sigmoid(reason_logits_latent) - factor_route_support.detach()).square(), valid)
    loss_escape = _masked_mean(escape_weight, valid)
    loss_propensity = _masked_mean((reason_propensity - 0.50).square(), pu_valid)
    total = (
        0.30 * loss_nll
        + 0.30 * loss_posterior
        + 0.08 * loss_rank
        + 0.05 * loss_factor_consistency
        + 0.01 * loss_escape
        + 0.01 * loss_propensity
    )
    return {
        "loss_reason_observation_nll": loss_nll,
        "loss_reason_posterior_bce": loss_posterior,
        "loss_reason_posterior_rank": loss_rank,
        "loss_reason_factor_latent_consistency": loss_factor_consistency,
        "loss_reason_escape": loss_escape,
        "loss_reason_propensity": loss_propensity,
        "loss_reason_selective_total": total,
        "pu_active_label_count": pu_valid.any(dim=0).to(dtype=reason_logits_latent.dtype).sum(),
    }
