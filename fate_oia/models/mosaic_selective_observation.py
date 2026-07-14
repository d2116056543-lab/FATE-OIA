from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class MOSAICSelectiveObservationModel(nn.Module):
    GROUPS = ("traffic_control", "obstacle", "lane", "other")

    def __init__(
        self,
        factor_names: Sequence[str],
        reason_observation: dict[int, Any],
        *,
        pi_min: float = 0.20,
        pi_max: float = 0.95,
        global_false_positive_max: float = 0.05,
    ) -> None:
        super().__init__()
        self.factor_names = tuple(factor_names)
        factor_index = {name: index for index, name in enumerate(self.factor_names)}
        if len(factor_index) != len(self.factor_names) or set(reason_observation) != set(range(21)):
            raise ValueError("selective observation requires unique factors and reasons 0..20")
        if not 0.0 < pi_min < pi_max < 1.0:
            raise ValueError("propensity bounds must satisfy 0 < pi_min < pi_max < 1")
        if not 0.0 <= global_false_positive_max <= 0.05:
            raise ValueError("global false-positive maximum must be in [0,0.05]")
        self.pi_min = float(pi_min)
        self.pi_max = float(pi_max)

        reason_factor_map = torch.zeros(21, len(self.factor_names))
        group_ids: list[int] = []
        false_positive_max: list[float] = []
        for reason_id in range(21):
            mapping = reason_observation[reason_id]
            group_ids.append(self.GROUPS.index(mapping["group"]))
            mapped_factors = set(mapping["support_factors"]) | set(mapping["visibility_factors"])
            if not mapped_factors:
                raise ValueError(f"reason {reason_id} has no observable factors for propensity")
            for factor_name in mapped_factors:
                if factor_name not in factor_index:
                    raise ValueError(f"reason {reason_id} references unknown factor {factor_name}")
                reason_factor_map[reason_id, factor_index[factor_name]] = 1.0
            label_max = float(mapping["false_positive_max"])
            if label_max > global_false_positive_max:
                raise ValueError("reason false-positive maximum exceeds the formal global cap")
            false_positive_max.append(label_max)
        reason_factor_map = reason_factor_map / reason_factor_map.sum(-1, keepdim=True).clamp_min(1.0)
        self.register_buffer("reason_factor_map", reason_factor_map, persistent=True)
        self.register_buffer("reason_group_ids", torch.tensor(group_ids, dtype=torch.long), persistent=True)
        self.register_buffer("false_positive_max", torch.tensor(false_positive_max), persistent=True)

        self.group_bias = nn.Parameter(torch.zeros(len(self.GROUPS)))
        self.raw_visibility_weight = nn.Parameter(torch.full((len(self.GROUPS),), _inverse_softplus(1.0)))
        self.raw_uncertainty_weight = nn.Parameter(torch.full((len(self.GROUPS),), _inverse_softplus(1.0)))
        self.false_positive_raw = nn.Parameter(torch.full((21,), math.log(0.2 / 0.8)))

    @property
    def reason_false_positive_rate(self) -> torch.Tensor:
        return self.false_positive_max * torch.sigmoid(self.false_positive_raw)

    @staticmethod
    def hide_observed_positives(
        observed_reason_targets: torch.Tensor,
        *,
        hide_fraction: float = 0.20,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0.0 <= hide_fraction <= 1.0:
            raise ValueError("hide_fraction must be in [0,1]")
        positive = observed_reason_targets > 0.5
        random_values = torch.rand(
            observed_reason_targets.shape,
            device=observed_reason_targets.device,
            generator=generator,
        )
        hidden_mask = positive & (random_values < hide_fraction)
        hidden_targets = observed_reason_targets.clone()
        hidden_targets[hidden_mask] = 0.0
        return hidden_targets, hidden_mask

    def _propensity(
        self,
        factor_visibility: torch.Tensor,
        factor_uncertainty: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        visibility_summary = torch.einsum(
            "rf,bf->br", self.reason_factor_map, factor_visibility.detach()
        )
        uncertainty_summary = torch.einsum(
            "rf,bf->br", self.reason_factor_map, factor_uncertainty.detach()
        )
        group_bias = self.group_bias[self.reason_group_ids]
        visibility_weight = F.softplus(self.raw_visibility_weight)[self.reason_group_ids]
        uncertainty_weight = F.softplus(self.raw_uncertainty_weight)[self.reason_group_ids]
        propensity_logit = (
            group_bias.unsqueeze(0)
            + visibility_weight.unsqueeze(0) * visibility_summary
            - uncertainty_weight.unsqueeze(0) * uncertainty_summary
        )
        propensity = self.pi_min + (self.pi_max - self.pi_min) * torch.sigmoid(propensity_logit)
        return propensity, visibility_summary, uncertainty_summary

    def forward(
        self,
        reason_logits_latent: torch.Tensor,
        observed_reason_targets: torch.Tensor,
        factor_visibility: torch.Tensor,
        factor_uncertainty: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compatibility wrapper; formal inference uses ``forward_inference``."""
        output = self.forward_inference(reason_logits_latent, factor_visibility, factor_uncertainty)
        output.update(self.posterior_from_observed_targets(reason_logits_latent, observed_reason_targets, output))
        return output

    def forward_inference(
        self,
        reason_logits_latent: torch.Tensor,
        factor_visibility: torch.Tensor,
        factor_uncertainty: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Inference-safe observation likelihood with no reason-label input."""
        batch_size = reason_logits_latent.shape[0]
        if tuple(reason_logits_latent.shape) != (batch_size, 21):
            raise ValueError("selective observation expects latent logits [B,21]")
        factor_shape = (batch_size, len(self.factor_names))
        if tuple(factor_visibility.shape) != factor_shape or tuple(factor_uncertainty.shape) != factor_shape:
            raise ValueError("selective observation factor summaries have invalid shapes")

        propensity, visibility_summary, uncertainty_summary = self._propensity(
            factor_visibility,
            factor_uncertainty,
        )
        epsilon = self.reason_false_positive_rate
        latent_probability = torch.sigmoid(reason_logits_latent)
        observation_probability = (
            propensity * latent_probability + epsilon.unsqueeze(0) * (1.0 - latent_probability)
        )

        return {
            "reason_propensity": propensity,
            "reason_false_positive_rate": epsilon,
            "reason_observation_prob": observation_probability,
            "propensity_visibility_summary": visibility_summary,
            "propensity_uncertainty_summary": uncertainty_summary,
            "propensity_visibility_slopes": F.softplus(self.raw_visibility_weight),
            "propensity_uncertainty_slopes": F.softplus(self.raw_uncertainty_weight),
            "propensity_pi_min": propensity.new_tensor(self.pi_min),
            "propensity_pi_max": propensity.new_tensor(self.pi_max),
            "reason_false_positive_max": self.false_positive_max,
        }

    @staticmethod
    def posterior_from_observed_targets(
        reason_logits_latent: torch.Tensor,
        observed_reason_targets: torch.Tensor,
        observation_output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Training/metric-only PU posterior; labels never enter inference forward."""
        if reason_logits_latent.shape != observed_reason_targets.shape or reason_logits_latent.shape[-1] != 21:
            raise ValueError("selective-observation posterior requires aligned [B,21] tensors")
        propensity = observation_output["reason_propensity"].detach().clamp(1e-7, 1.0 - 1e-7)
        epsilon = observation_output["reason_false_positive_rate"].detach().clamp(1e-7, 1.0 - 1e-7)
        if epsilon.ndim == 1:
            epsilon = epsilon.unsqueeze(0)
        log_hidden_positive = F.logsigmoid(reason_logits_latent.detach()) + torch.log1p(-propensity)
        log_true_negative = F.logsigmoid(-reason_logits_latent.detach()) + torch.log1p(-epsilon)
        posterior_zero = torch.exp(log_hidden_positive - torch.logaddexp(log_hidden_positive, log_true_negative))
        posterior = torch.where(observed_reason_targets > 0.5, torch.ones_like(posterior_zero), posterior_zero)
        return {"reason_latent_posterior": posterior.detach()}
