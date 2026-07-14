from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class MOSAICICDORObservationHead(nn.Module):
    """Inference-safe selective-observation process for latent reason logits.

    ``forward`` deliberately has no labels.  Posterior recovery is a trainer
    loss operation through ``posterior_from_observed_targets`` so test forward
    cannot consume observed reason labels.
    """

    GROUPS = ("traffic_control", "obstacle", "lane", "other")

    def __init__(self, ontology: dict[str, Any], *, pi_min: float = 0.20, pi_max: float = 0.95) -> None:
        super().__init__()
        if not 0.0 < pi_min < pi_max < 1.0:
            raise ValueError("IC-DOR observation propensity bounds must satisfy 0 < min < max < 1")
        factors = ontology["factors"]
        factor_index = ontology["factor_index"]
        routes = ontology["reason_routes"]
        if set(routes) != set(range(21)):
            raise ValueError("IC-DOR observation head requires all 21 reason routes")
        reason_factor_map = torch.zeros(21, len(factors))
        group_ids: list[int] = []
        for reason_index in range(21):
            route = routes[reason_index]
            group_ids.append(self.GROUPS.index(route["group"]))
            allowed = set(route["direct_factors"]) | set(route["latent_factors"])
            if not allowed:
                raise ValueError(f"IC-DOR reason {reason_index} must expose at least one observable factor")
            for factor_name in allowed:
                reason_factor_map[reason_index, factor_index[factor_name]] = 1.0
        reason_factor_map = reason_factor_map / reason_factor_map.sum(dim=-1, keepdim=True)
        self.register_buffer("reason_factor_map", reason_factor_map, persistent=True)
        self.register_buffer("reason_group_ids", torch.tensor(group_ids, dtype=torch.long), persistent=True)
        self.pi_min = float(pi_min)
        self.pi_max = float(pi_max)
        self.group_bias = nn.Parameter(torch.zeros(len(self.GROUPS)))
        self.raw_visibility_weight = nn.Parameter(torch.full((len(self.GROUPS),), _inverse_softplus(1.0)))
        self.raw_uncertainty_weight = nn.Parameter(torch.full((len(self.GROUPS),), _inverse_softplus(1.0)))
        self.false_positive_raw = nn.Parameter(torch.full((21,), math.log(0.2 / 0.8)))
        self.register_buffer("false_positive_max", torch.full((21,), 0.05), persistent=True)

    @property
    def false_positive_rate(self) -> torch.Tensor:
        return self.false_positive_max * torch.sigmoid(self.false_positive_raw)

    def _propensity(self, factor_visibility: torch.Tensor, factor_uncertainty: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visibility = torch.einsum("rf,bf->br", self.reason_factor_map, factor_visibility.detach())
        uncertainty = torch.einsum("rf,bf->br", self.reason_factor_map, factor_uncertainty.detach())
        group_bias = self.group_bias[self.reason_group_ids]
        visibility_weight = F.softplus(self.raw_visibility_weight)[self.reason_group_ids]
        uncertainty_weight = F.softplus(self.raw_uncertainty_weight)[self.reason_group_ids]
        logits = group_bias.unsqueeze(0) + visibility_weight.unsqueeze(0) * visibility - uncertainty_weight.unsqueeze(0) * uncertainty
        propensity = self.pi_min + (self.pi_max - self.pi_min) * torch.sigmoid(logits)
        return propensity, visibility, uncertainty

    def forward(
        self,
        reason_logits_latent: torch.Tensor,
        factor_visibility: torch.Tensor,
        factor_uncertainty: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = reason_logits_latent.shape[0]
        if tuple(reason_logits_latent.shape) != (batch_size, 21):
            raise ValueError("IC-DOR observation head requires latent reason logits [B,21]")
        if factor_visibility.shape != factor_uncertainty.shape or factor_visibility.shape[0] != batch_size:
            raise ValueError("IC-DOR observation head factor summaries must be [B,F]")
        if factor_visibility.shape[1] != self.reason_factor_map.shape[1]:
            raise ValueError("IC-DOR observation head received the wrong factor count")
        propensity, visibility, uncertainty = self._propensity(factor_visibility, factor_uncertainty)
        epsilon = self.false_positive_rate.unsqueeze(0)
        latent_probability = torch.sigmoid(reason_logits_latent)
        observed_probability = propensity * latent_probability + epsilon * (1.0 - latent_probability)
        return {
            "reason_propensity": propensity,
            "reason_false_positive_rate": epsilon.expand_as(propensity),
            "reason_observation_prob": observed_probability,
            "reason_observation_logits": torch.logit(observed_probability.clamp(1e-6, 1.0 - 1e-6)),
            "propensity_visibility_summary": visibility,
            "propensity_uncertainty_summary": uncertainty,
            "propensity_visibility_slopes": F.softplus(self.raw_visibility_weight),
            "propensity_uncertainty_slopes": F.softplus(self.raw_uncertainty_weight),
        }

    @staticmethod
    def posterior_from_observed_targets(
        reason_logits_latent: torch.Tensor,
        observed_reason_targets: torch.Tensor,
        observation_output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if reason_logits_latent.shape != observed_reason_targets.shape or reason_logits_latent.shape[-1] != 21:
            raise ValueError("IC-DOR posterior requires matching [B,21] latent logits and observed targets")
        propensity = observation_output["reason_propensity"].detach().clamp(1e-6, 1.0 - 1e-6)
        epsilon = observation_output["reason_false_positive_rate"].detach().clamp(1e-6, 1.0 - 1e-6)
        log_latent_positive = F.logsigmoid(reason_logits_latent.detach())
        log_latent_negative = F.logsigmoid(-reason_logits_latent.detach())
        log_hidden_positive = log_latent_positive + torch.log1p(-propensity)
        log_true_negative = log_latent_negative + torch.log1p(-epsilon)
        posterior_zero = torch.exp(log_hidden_positive - torch.logaddexp(log_hidden_positive, log_true_negative))
        posterior = torch.where(observed_reason_targets > 0.5, torch.ones_like(posterior_zero), posterior_zero)
        return {"reason_latent_posterior": posterior.detach()}
