from __future__ import annotations

import torch
from torch import nn

from .tida_predicate_differential import finite_difference


TRANSITION_SCALE_NAMES = ("velocity", "acceleration", "region_velocity", "persistence")


def _zero_residual_projection(input_dim: int, output_dim: int) -> nn.Sequential:
    projection = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, output_dim),
        nn.GELU(),
        nn.Linear(output_dim, output_dim),
    )
    nn.init.zeros_(projection[-1].weight)
    nn.init.zeros_(projection[-1].bias)
    return projection


def _masked_recent_mean(values: torch.Tensor, valid: torch.Tensor, decay: float = 0.8) -> torch.Tensor:
    steps = values.shape[1]
    powers = torch.arange(steps - 1, -1, -1, device=values.device, dtype=values.dtype)
    weights = decay ** powers
    shape = (1, steps) + (1,) * (values.ndim - 2)
    expanded = weights.view(shape) * valid[(slice(None), slice(None)) + (None,) * (values.ndim - 2)].to(values.dtype)
    return (values * expanded).sum(1) / expanded.sum(1).clamp_min(1e-6)


def _endpoint_slope(values: torch.Tensor, timestamps: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    batch, steps = valid.shape
    indices = torch.arange(steps, device=values.device).view(1, steps).expand(batch, -1)
    first_index = indices.masked_fill(~valid, steps).min(1).values.clamp_max(steps - 1)
    last_index = indices.masked_fill(~valid, -1).max(1).values.clamp_min(0)
    batch_index = torch.arange(batch, device=values.device)
    first = values[batch_index, first_index]
    last = values[batch_index, last_index]
    delta_t = (timestamps[batch_index, last_index] - timestamps[batch_index, first_index]).clamp_min(1e-6)
    expand = (slice(None),) + (None,) * (values.ndim - 2)
    slope = (last - first) / delta_t[expand]
    enough = valid.sum(1) >= 2
    return slope * enough[expand].to(slope.dtype)


class TIDAFlowTransitionBank(nn.Module):
    """Compact signed motion factors built from ordered query trajectories."""

    def __init__(self, dim: int = 384, region_count: int = 5) -> None:
        super().__init__()
        self.dim = int(dim)
        self.region_count = int(region_count)
        feature_dim = self.dim * 2 + self.region_count + 1
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )
        self.scale_residuals = nn.ModuleList(
            [
                _zero_residual_projection(self.dim, self.dim),
                _zero_residual_projection(self.dim, self.dim),
                _zero_residual_projection(self.region_count, self.dim),
                _zero_residual_projection(1, self.dim),
            ]
        )
        self.scale_type_embedding = nn.Parameter(torch.zeros(len(TRANSITION_SCALE_NAMES), self.dim))

    def forward(
        self,
        query_trajectory: torch.Tensor,
        region_mass: torch.Tensor,
        timestamps: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if query_trajectory.ndim != 4:
            raise ValueError("query_trajectory must be [B,T,Q,D]")
        if region_mass.shape[:3] != query_trajectory.shape[:3]:
            raise ValueError("region_mass must agree with trajectory B,T,Q")
        if region_mass.shape[-1] != self.region_count:
            raise ValueError("region count mismatch")

        velocity_steps, velocity_valid = finite_difference(query_trajectory, timestamps, valid_mask)
        region_steps, _ = finite_difference(region_mass, timestamps, valid_mask)
        midpoint = 0.5 * (timestamps[:, 1:] + timestamps[:, :-1])
        acceleration_steps, acceleration_valid = finite_difference(velocity_steps, midpoint, velocity_valid)

        # Endpoint displacement is strictly antisymmetric under sequence
        # reversal even for the quadratic, non-uniform frame timestamps used
        # by TIDA. Recent finite differences remain useful for acceleration
        # and persistence, but must not define the primary signed direction.
        velocity = _endpoint_slope(query_trajectory, timestamps, valid_mask)
        acceleration = _masked_recent_mean(acceleration_steps, acceleration_valid)
        region_velocity = _endpoint_slope(region_mass, timestamps, valid_mask)

        direction = torch.sign(velocity_steps)
        reference = torch.sign(velocity)[:, None]
        agreement = (direction == reference).to(query_trajectory.dtype)
        moving = velocity_steps.abs() > 1e-6
        valid = velocity_valid[:, :, None, None] & moving
        persistence = (agreement * valid).sum((1, 3)) / valid.sum((1, 3)).clamp_min(1)

        features = torch.cat([velocity, acceleration, region_velocity, persistence[..., None]], dim=-1)
        legacy_transition_tokens = self.projection(features)
        valid_fraction = velocity_valid.to(query_trajectory.dtype).mean(1, keepdim=True)
        motion_strength = velocity.norm(dim=-1) + 0.5 * acceleration.norm(dim=-1) + region_velocity.norm(dim=-1)
        reliability = valid_fraction * torch.tanh(motion_strength)
        scale_inputs = (velocity, acceleration, region_velocity, persistence[..., None])
        scale_residual = torch.stack(
            [projection(value) for projection, value in zip(self.scale_residuals, scale_inputs)],
            dim=2,
        )
        transition_tokens_by_scale = (
            legacy_transition_tokens[:, :, None]
            + scale_residual
            + self.scale_type_embedding[None, None]
        )
        transition_tokens = transition_tokens_by_scale.mean(2)
        history_available = valid_mask.sum(1) >= 2
        motion_salience = valid_fraction * torch.log1p(motion_strength)
        motion_salience = motion_salience * history_available[:, None].to(motion_salience.dtype)
        transition_consistency = valid_fraction * persistence.clamp(0.0, 1.0)
        transition_consistency = transition_consistency * history_available[:, None].to(transition_consistency.dtype)
        return {
            "transition_tokens": transition_tokens,
            "legacy_transition_tokens": legacy_transition_tokens,
            "transition_tokens_by_scale": transition_tokens_by_scale,
            "transition_scale_names": TRANSITION_SCALE_NAMES,
            "velocity": velocity,
            "acceleration": acceleration,
            "region_velocity": region_velocity,
            "persistence": persistence,
            "transition_reliability": reliability.clamp(0.0, 1.0),
            "motion_salience": motion_salience,
            "transition_consistency": transition_consistency,
            "history_available": history_available,
            "velocity_step_valid": velocity_valid,
            "acceleration_step_valid": acceleration_valid,
        }
