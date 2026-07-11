from __future__ import annotations

import torch
from torch.nn import functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape != mask.shape:
        raise ValueError("reason loss values and mask must have matching shapes")
    weights = mask.to(dtype=values.dtype)
    denominator = weights.sum()
    loss = (values * weights).sum() / denominator.clamp_min(1e-12)
    return torch.where(denominator > 0, loss, _zero(values)), (mask > 0).sum().detach()


def fixed_propensity_observation_loss(
    observation_probability: torch.Tensor,
    observed_reason_targets: torch.Tensor,
) -> torch.Tensor:
    """Observed-label likelihood without converting an unobserved reason into a latent negative."""
    if observation_probability.shape != observed_reason_targets.shape:
        raise ValueError("observation probability and observed reasons must have matching shapes")
    probability = observation_probability.float().clamp(1e-7, 1.0 - 1e-7)
    return F.binary_cross_entropy_with_logits(
        torch.logit(probability), observed_reason_targets.float()
    )


def build_mosaic_reason_loss(
    reason_logits_latent: torch.Tensor,
    observed_reason_targets: torch.Tensor,
    observation_probability: torch.Tensor,
    posterior_detached: torch.Tensor,
    posterior_live: torch.Tensor,
    propensity: torch.Tensor,
    synthetic_hidden_positive_mask: torch.Tensor,
    *,
    propensity_visibility_slopes: torch.Tensor | None = None,
    propensity_uncertainty_slopes: torch.Tensor | None = None,
    propensity_pi_min: torch.Tensor | float = 0.20,
    propensity_pi_max: torch.Tensor | float = 0.95,
    reason_false_positive_rate: torch.Tensor | None = None,
    reason_false_positive_max: torch.Tensor | None = None,
    rank_loss: torch.Tensor | None = None,
    prevalence_observed_targets: torch.Tensor | None = None,
    posterior_bce_weight: float = 0.30,
    posterior_rank_weight: float = 0.08,
    missing_recovery_weight: float = 0.10,
    latent_rate_range_weight: float = 0.03,
    propensity_regularization_weight: float = 0.01,
) -> dict[str, torch.Tensor]:
    if any(
        weight < 0
        for weight in (
            posterior_bce_weight,
            posterior_rank_weight,
            missing_recovery_weight,
            latent_rate_range_weight,
            propensity_regularization_weight,
        )
    ):
        raise ValueError("reason loss weights must be non-negative")
    expected_shape = reason_logits_latent.shape
    tensors = (
        observed_reason_targets,
        observation_probability,
        posterior_detached,
        posterior_live,
        propensity,
        synthetic_hidden_positive_mask,
    )
    if any(tensor.shape != expected_shape for tensor in tensors):
        raise ValueError("all MOSAIC reason loss tensors must have matching [B,R] shapes")
    if posterior_detached.requires_grad:
        raise ValueError("M-step posterior target must be detached")

    observed = observed_reason_targets.to(dtype=reason_logits_latent.dtype)
    observation_nll = fixed_propensity_observation_loss(observation_probability, observed)
    posterior_bce = F.binary_cross_entropy_with_logits(reason_logits_latent, posterior_detached)
    if rank_loss is None:
        rank_loss = _zero(reason_logits_latent)
    if rank_loss.ndim != 0:
        raise ValueError("posterior rank loss must be scalar")

    hidden_mask = synthetic_hidden_positive_mask.to(dtype=torch.bool)
    recovery_values = -torch.log(posterior_live.clamp_min(1e-7))
    missing_recovery, count_missing_recovery = _masked_mean(recovery_values, hidden_mask)

    latent_rate = torch.sigmoid(reason_logits_latent).mean(dim=0)
    if prevalence_observed_targets is None:
        prevalence_observed_targets = observed
    if prevalence_observed_targets.shape != expected_shape:
        raise ValueError("prevalence observed targets must match reason logits")
    observed_rate = prevalence_observed_targets.to(dtype=reason_logits_latent.dtype).mean(dim=0).detach()
    upper_rate = torch.minimum(3.0 * observed_rate + 0.02, observed_rate.new_full((), 0.60))
    latent_rate_range = (
        F.relu(observed_rate - latent_rate).square()
        + F.relu(latent_rate - upper_rate).square()
    ).mean()

    if propensity_visibility_slopes is None or propensity_uncertainty_slopes is None:
        raise ValueError("propensity regularization requires both learned group slopes")
    if propensity_visibility_slopes.shape != propensity_uncertainty_slopes.shape:
        raise ValueError("propensity slope tensors must have matching shapes")
    slope_regularization = (
        propensity_visibility_slopes.square() + propensity_uncertainty_slopes.square()
    ).mean()
    pi_min = torch.as_tensor(propensity_pi_min, device=propensity.device, dtype=propensity.dtype)
    pi_max = torch.as_tensor(propensity_pi_max, device=propensity.device, dtype=propensity.dtype)
    boundary_regularization = (
        F.relu(0.05 - (propensity - pi_min)).square()
        + F.relu(0.05 - (pi_max - propensity)).square()
    ).mean()
    if reason_false_positive_rate is None or reason_false_positive_max is None:
        raise ValueError("propensity regularization requires false-positive rates and maxima")
    if reason_false_positive_rate.shape != reason_false_positive_max.shape:
        raise ValueError("false-positive rates and maxima must have matching shapes")
    epsilon_ratio = torch.where(
        reason_false_positive_max > 0,
        reason_false_positive_rate / reason_false_positive_max.clamp_min(1e-12),
        torch.zeros_like(reason_false_positive_rate),
    )
    epsilon_regularization = epsilon_ratio.square().mean()
    propensity_regularization = (
        slope_regularization
        + 0.10 * boundary_regularization
        + 0.10 * epsilon_regularization
    )

    total = (
        observation_nll
        + posterior_bce_weight * posterior_bce
        + posterior_rank_weight * rank_loss
        + missing_recovery_weight * missing_recovery
        + latent_rate_range_weight * latent_rate_range
        + propensity_regularization_weight * propensity_regularization
    )
    count_all = reason_logits_latent.new_tensor(reason_logits_latent.numel(), dtype=torch.long)
    return {
        "loss_observation_nll": observation_nll,
        "loss_posterior_bce": posterior_bce,
        "loss_posterior_rank": rank_loss,
        "loss_missing_recovery": missing_recovery,
        "loss_latent_rate_range": latent_rate_range,
        "loss_propensity_regularization": propensity_regularization,
        "loss_propensity_slope": slope_regularization,
        "loss_propensity_boundary": boundary_regularization,
        "loss_propensity_epsilon": epsilon_regularization,
        "loss_reason_total": total,
        "count_observation_nll": count_all,
        "count_posterior_bce": count_all,
        "count_posterior_rank": count_all,
        "count_missing_recovery": count_missing_recovery,
        "count_latent_rate_range": reason_logits_latent.new_tensor(
            reason_logits_latent.shape[1], dtype=torch.long
        ),
        "count_propensity_regularization": count_all,
    }
