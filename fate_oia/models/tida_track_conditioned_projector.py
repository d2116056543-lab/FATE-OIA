from __future__ import annotations

import math

import torch
from torch import nn


class TIDATrackConditionedProjector(nn.Module):
    """Convert tracked patch trajectories into target-private frame tokens."""

    def __init__(
        self,
        dim: int,
        num_targets: int,
        *,
        attention_temperature: float = 1.0,
        attention_topk: int | None = None,
        confidence_power: float = 0.0,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_targets <= 0:
            raise ValueError("invalid track-conditioned projector dimensions")
        if attention_temperature <= 0:
            raise ValueError("attention_temperature must be positive")
        if attention_topk is not None and attention_topk <= 0:
            raise ValueError("attention_topk must be positive when provided")
        if confidence_power < 0:
            raise ValueError("confidence_power must be non-negative")
        self.dim = int(dim)
        self.num_targets = int(num_targets)
        self.attention_temperature = float(attention_temperature)
        self.attention_topk = None if attention_topk is None else int(attention_topk)
        self.confidence_power = float(confidence_power)
        self.query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False))
        self.key = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False))
        self.value = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim, bias=False))
        self.geometry = nn.Sequential(
            nn.Linear(8, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.output = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim, bias=False),
        )

    @staticmethod
    def _motion_features(
        xy: torch.Tensor, exclusive_displacement: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        velocity = torch.zeros_like(xy)
        velocity[..., 1:, :] = exclusive_displacement
        acceleration = torch.zeros_like(xy)
        if xy.shape[-2] > 2:
            acceleration[..., 2:, :] = (
                exclusive_displacement[..., 1:, :]
                - exclusive_displacement[..., :-1, :]
            )
        speed = velocity.square().sum(-1, keepdim=True).sqrt()
        radial = (velocity * xy).sum(-1, keepdim=True)
        geometry = torch.cat((xy, velocity, acceleration, speed, radial), dim=-1)
        return geometry, speed

    def _pool(
        self,
        target_tokens: torch.Tensor,
        appearance: torch.Tensor,
        xy: torch.Tensor,
        visibility: torch.Tensor,
        exclusive_displacement: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch, groups, _, _, dim = appearance.shape
        labels = target_tokens.shape[1]
        geometry, speed = self._motion_features(xy, exclusive_displacement)
        if groups == 1:
            appearance = appearance.expand(-1, labels, -1, -1, -1)
            geometry = geometry.expand(-1, labels, -1, -1, -1)
            visibility = visibility.expand(-1, labels, -1, -1)
            speed = speed.expand(-1, labels, -1, -1, -1)

        query = self.query(target_tokens)
        key = self.key(appearance)
        score = torch.einsum("bld,blktd->bltk", query, key) / (
            math.sqrt(dim) * self.attention_temperature
        )
        confidence = visibility.permute(0, 1, 3, 2).clamp(0.0, 1.0)
        valid = confidence > 0
        if self.confidence_power > 0:
            score = score + self.confidence_power * confidence.clamp_min(1e-6).log()
        if self.attention_topk is not None and self.attention_topk < score.shape[-1]:
            masked_score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
            topk_index = masked_score.topk(self.attention_topk, dim=-1).indices
            topk_mask = torch.zeros_like(valid)
            topk_mask.scatter_(-1, topk_index, True)
            valid = valid & topk_mask
        safe_valid = valid.clone()
        no_track = ~safe_valid.any(-1)
        safe_valid[..., 0] |= no_track
        score = score.masked_fill(~safe_valid, torch.finfo(score.dtype).min)
        attention = score.softmax(-1).masked_fill(~valid, 0.0)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)

        value = self.value(appearance) + self.geometry(geometry)
        pooled = torch.einsum("bltk,blktd->bltd", attention, value)
        available = valid.any(-1).to(pooled.dtype)
        pooled = self.output(pooled) * available[..., None]
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
        effective_count = entropy.exp() * available
        motion_rms = (
            torch.einsum(
                "bltk,bltk->bl",
                attention,
                speed.squeeze(-1).permute(0, 1, 3, 2).square(),
            )
            / available.sum(-1).clamp_min(1.0)
        ).sqrt()
        return {
            "tokens": pooled.permute(0, 2, 1, 3),
            "attention": attention,
            "available": available,
            "effective_count": effective_count,
            "motion_rms": motion_rms,
            "attention_entropy": entropy,
        }

    def forward(
        self,
        target_tokens: torch.Tensor,
        trajectory_appearance: torch.Tensor,
        trajectory_xy: torch.Tensor,
        trajectory_visibility: torch.Tensor,
        trajectory_exclusive_displacement: torch.Tensor,
        *,
        action_specific: bool,
    ) -> dict[str, torch.Tensor]:
        if target_tokens.ndim != 3:
            raise ValueError("target_tokens must be [B,L,D]")
        if trajectory_appearance.ndim != 5:
            raise ValueError("trajectory_appearance must be [B,G,K,T,D]")
        batch, groups, tracks, frames, dim = trajectory_appearance.shape
        labels = target_tokens.shape[1]
        if target_tokens.shape != (batch, labels, dim) or dim != self.dim:
            raise ValueError("target token dimensions do not match trajectories")
        if labels != self.num_targets:
            raise ValueError("target count does not match projector")
        expected_xy = (batch, groups, tracks, frames, 2)
        if trajectory_xy.shape != expected_xy:
            raise ValueError("trajectory_xy shape mismatch")
        if trajectory_visibility.shape != expected_xy[:-1]:
            raise ValueError("trajectory_visibility shape mismatch")
        if trajectory_exclusive_displacement.shape != (
            batch, groups, tracks, frames - 1, 2
        ):
            raise ValueError("exclusive displacement shape mismatch")
        if action_specific and groups != labels:
            raise ValueError("action-specific trajectories require G=L")
        if not action_specific and groups != 1:
            raise ValueError("shared semantic trajectories require G=1")

        ordered = self._pool(
            target_tokens,
            trajectory_appearance,
            trajectory_xy,
            trajectory_visibility,
            trajectory_exclusive_displacement,
        )
        reversed_xy = trajectory_xy.flip(3)
        reversed_exclusive = reversed_xy[..., 1:, :] - reversed_xy[..., :-1, :]
        shuffled = self._pool(
            target_tokens,
            trajectory_appearance.flip(3),
            reversed_xy,
            trajectory_visibility.flip(3),
            reversed_exclusive,
        )
        return {
            "track_conditioned_tokens": ordered["tokens"],
            "track_shuffled_conditioned_tokens": shuffled["tokens"],
            "track_attention": ordered["attention"],
            "track_available": ordered["available"],
            "track_effective_count": ordered["effective_count"],
            "track_motion_rms": ordered["motion_rms"],
            "track_attention_entropy": ordered["attention_entropy"],
        }
