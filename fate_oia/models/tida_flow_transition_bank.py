from __future__ import annotations

import torch
from torch import nn

from .tida_predicate_differential import finite_difference


def _masked_recent_mean(values: torch.Tensor, valid: torch.Tensor, decay: float = 0.8) -> torch.Tensor:
    steps = values.shape[1]
    powers = torch.arange(steps - 1, -1, -1, device=values.device, dtype=values.dtype)
    weights = decay ** powers
    shape = (1, steps) + (1,) * (values.ndim - 2)
    expanded = weights.view(shape) * valid[(slice(None), slice(None)) + (None,) * (values.ndim - 2)].to(values.dtype)
    return (values * expanded).sum(1) / expanded.sum(1).clamp_min(1e-6)


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

        velocity = _masked_recent_mean(velocity_steps, velocity_valid)
        acceleration = _masked_recent_mean(acceleration_steps, acceleration_valid)
        region_velocity = _masked_recent_mean(region_steps, velocity_valid)

        direction = torch.sign(velocity_steps)
        reference = torch.sign(velocity)[:, None]
        agreement = (direction == reference).to(query_trajectory.dtype)
        moving = velocity_steps.abs() > 1e-6
        valid = velocity_valid[:, :, None, None] & moving
        persistence = (agreement * valid).sum((1, 3)) / valid.sum((1, 3)).clamp_min(1)

        features = torch.cat([velocity, acceleration, region_velocity, persistence[..., None]], dim=-1)
        transition_tokens = self.projection(features)
        valid_fraction = velocity_valid.to(query_trajectory.dtype).mean(1, keepdim=True)
        motion_strength = velocity.norm(dim=-1) + 0.5 * acceleration.norm(dim=-1) + region_velocity.norm(dim=-1)
        reliability = valid_fraction * torch.tanh(motion_strength)
        return {
            "transition_tokens": transition_tokens,
            "velocity": velocity,
            "acceleration": acceleration,
            "region_velocity": region_velocity,
            "persistence": persistence,
            "transition_reliability": reliability.clamp(0.0, 1.0),
            "velocity_step_valid": velocity_valid,
            "acceleration_step_valid": acceleration_valid,
        }
