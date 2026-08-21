from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
import yaml
from torch import nn


ROLE_NAMES = ("static_anchor", "dynamic_actor", "terminal_context")


def load_predicate_roles(path: str | Path, predicate_names: Sequence[str]) -> dict[str, list[str]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    roles = {role: [str(name) for name in payload.get(role, [])] for role in ROLE_NAMES}
    flattened = [name for role in ROLE_NAMES for name in roles[role]]
    expected = list(predicate_names)
    duplicates = sorted({name for name in flattened if flattened.count(name) > 1})
    missing = sorted(set(expected) - set(flattened))
    extra = sorted(set(flattened) - set(expected))
    if duplicates or missing or extra or len(flattened) != len(expected):
        raise ValueError(f"predicate role mapping is not an exact cover: missing={missing}, extra={extra}, duplicate={duplicates}")
    return roles


def finite_difference(
    values: torch.Tensor, timestamps: torch.Tensor, frame_valid: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    if values.shape[:2] != timestamps.shape or timestamps.shape != frame_valid.shape:
        raise ValueError("values, timestamps, and frame_valid time dimensions must agree")
    delta_t = (timestamps[:, 1:] - timestamps[:, :-1]).clamp_min(eps)
    expand = (slice(None), slice(None)) + (None,) * (values.ndim - 2)
    derivative = (values[:, 1:] - values[:, :-1]) / delta_t[expand]
    pair_valid = frame_valid[:, 1:] & frame_valid[:, :-1]
    derivative = derivative * pair_valid[expand].to(derivative.dtype)
    return derivative, pair_valid


def robust_common_motion(velocity: torch.Tensor, static_mask: torch.Tensor, huber_delta: float = 1.0) -> torch.Tensor:
    if static_mask.dtype != torch.bool or static_mask.numel() != velocity.shape[2]:
        raise ValueError("static_mask must select predicate dimension")
    static = velocity[:, :, static_mask]
    if static.shape[2] == 0:
        raise ValueError("at least one static anchor is required")
    median = static.median(dim=2, keepdim=True).values
    residual = (static - median).norm(dim=-1)
    weights = torch.where(residual <= huber_delta, torch.ones_like(residual), huber_delta / residual.clamp_min(1e-6))
    center = (weights[..., None] * static).sum(2) / weights.sum(2, keepdim=True).clamp_min(1e-6)
    return center.detach()


def _ema_last(values: torch.Tensor, valid: torch.Tensor, decay: float = 0.8) -> torch.Tensor:
    steps = values.shape[1]
    powers = torch.arange(steps - 1, -1, -1, device=values.device, dtype=values.dtype)
    weights = decay**powers
    shape = (1, steps) + (1,) * (values.ndim - 2)
    weights = weights.view(shape) * valid[(slice(None), slice(None)) + (None,) * (values.ndim - 2)].to(values.dtype)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1e-6)


class TIDAPredicateDifferential(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        predicate_names: Sequence[str] | None = None,
        roles: dict[str, list[str]] | None = None,
        role_path: str | Path = "configs/tida_predicate_roles.yaml",
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.predicate_names = list(predicate_names or [f"p{i}" for i in range(32)])
        self.roles = roles or load_predicate_roles(role_path, self.predicate_names)
        flattened = [name for role in ROLE_NAMES for name in self.roles.get(role, [])]
        if sorted(flattened) != sorted(self.predicate_names) or len(flattened) != len(set(flattened)):
            raise ValueError("roles must exactly cover predicate_names")
        static = torch.tensor([name in self.roles["static_anchor"] for name in self.predicate_names], dtype=torch.bool)
        role_ids = torch.tensor([next(i for i, role in enumerate(ROLE_NAMES) if name in self.roles[role]) for name in self.predicate_names])
        self.register_buffer("static_mask", static, persistent=True)
        self.register_buffer("role_ids", role_ids, persistent=True)
        self.role_embedding = nn.Embedding(len(ROLE_NAMES), dim)
        # Predicate-specific diagonal A_p. Identity initialization removes
        # shared camera motion before learning semantic-specific adjustment.
        self.common_projection_raw = nn.Parameter(torch.zeros(len(self.predicate_names), dim))
        input_dim = dim * 4 + 5 * 2 + 2
        self.state_projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(
        self,
        history_predicate_tokens: torch.Tensor,
        target_predicate_tokens: torch.Tensor,
        predicate_innovation: torch.Tensor,
        timestamps: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        history_region_mass: torch.Tensor,
        target_region_mass: torch.Tensor,
        reliability: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        trajectory = torch.cat([history_predicate_tokens, target_predicate_tokens[:, None]], dim=1)
        region_mass = torch.cat([history_region_mass, target_region_mass[:, None]], dim=1)
        velocity, velocity_valid = finite_difference(trajectory, timestamps, frame_valid_mask)
        region_velocity, _ = finite_difference(region_mass, timestamps, frame_valid_mask)
        midpoint = 0.5 * (timestamps[:, 1:] + timestamps[:, :-1])
        acceleration, acceleration_valid = finite_difference(velocity, midpoint, velocity_valid)
        region_acceleration, _ = finite_difference(region_velocity, midpoint, velocity_valid)
        common = robust_common_motion(velocity, self.static_mask)
        common_scale = 1.0 + 0.25 * torch.tanh(self.common_projection_raw)
        projected_common = common[:, :, None] * common_scale[None, None]
        relative_velocity = velocity - projected_common
        relative_velocity_ema = _ema_last(relative_velocity, velocity_valid)
        acceleration_ema = _ema_last(acceleration, acceleration_valid)
        region_velocity_ema = _ema_last(region_velocity, velocity_valid)
        region_acceleration_ema = _ema_last(region_acceleration, acceleration_valid)
        cosine = F.cosine_similarity(trajectory[:, 1:], trajectory[:, :-1], dim=-1).clamp(-1, 1)
        persistence = (cosine * velocity_valid[:, :, None]).sum(1) / velocity_valid.sum(1, keepdim=True).clamp_min(1)
        persistence = 0.5 * (persistence + 1.0)
        state_input = torch.cat(
            [
                target_predicate_tokens,
                predicate_innovation,
                relative_velocity_ema,
                acceleration_ema,
                region_velocity_ema,
                region_acceleration_ema,
                persistence[..., None],
                reliability[..., None],
            ],
            dim=-1,
        )
        state = self.state_projection(state_input)
        routing_state = state + self.role_embedding(self.role_ids)[None]
        return {
            "predicate_differential_state": state,
            "predicate_routing_key_state": routing_state,
            "predicate_velocity": relative_velocity_ema,
            "predicate_acceleration": acceleration_ema,
            "predicate_velocity_norm": relative_velocity_ema.norm(dim=-1),
            "predicate_acceleration_norm": acceleration_ema.norm(dim=-1),
            "predicate_persistence": persistence,
            "predicate_region_mass": region_mass,
            "predicate_region_mass_velocity": region_velocity_ema,
            "predicate_region_mass_acceleration": region_acceleration_ema,
            "common_motion": common,
            "common_motion_norm": common.norm(dim=-1).mean(1),
            "predicate_role_ids": self.role_ids,
        }
