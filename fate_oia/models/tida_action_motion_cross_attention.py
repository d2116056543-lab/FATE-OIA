from __future__ import annotations

import math

import torch
from torch import nn


class TIDAActionMotionCrossAttention(nn.Module):
    """Read ordered action-token motion without feeding reason evidence into action."""

    def __init__(self, dim: int = 384, num_actions: int = 4, cap: float = 0.15) -> None:
        super().__init__()
        if dim <= 0 or num_actions <= 0 or cap <= 0:
            raise ValueError("dim, num_actions, and cap must be positive")
        self.dim = int(dim)
        self.num_actions = int(num_actions)
        self.cap = float(cap)
        self.action_identity = nn.Parameter(torch.randn(num_actions, dim) * 0.02)
        self.motion_projection = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.patch_motion_projection = nn.Sequential(
            nn.LayerNorm(dim + 5),
            nn.Linear(dim + 5, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        self.value_projection = nn.Linear(dim, dim, bias=False)
        self.logit_context = nn.Sequential(nn.Linear(2, dim), nn.Tanh())
        self.readout = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU())
        self.output = nn.Linear(dim, 1)
        self.gate = nn.Linear(dim, 1)
        self.same_action_bias = nn.Parameter(torch.tensor(0.5))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)

    @staticmethod
    def _safe_attention(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        masked = scores.masked_fill(~valid, -torch.inf)
        has_history = valid.any(-1, keepdim=True)
        masked = torch.where(has_history, masked, torch.zeros_like(masked))
        attention = torch.softmax(masked, dim=-1)
        return torch.where(has_history, attention, torch.zeros_like(attention))

    def forward(
        self,
        action_nodes: torch.Tensor,
        history_action_tokens: torch.Tensor,
        timestamps: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        base_logits: torch.Tensor,
        *,
        patch_tokens: torch.Tensor | None = None,
        patch_xy: torch.Tensor | None = None,
        patch_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if action_nodes.ndim != 3 or action_nodes.shape[1:] != (self.num_actions, self.dim):
            raise ValueError("action_nodes must have shape [B,A,D]")
        if history_action_tokens.ndim != 4 or history_action_tokens.shape[2:] != (self.num_actions, self.dim):
            raise ValueError("history_action_tokens must have shape [B,T,A,D]")
        batch, frames = history_action_tokens.shape[:2]
        if timestamps.shape != (batch, frames) or frame_valid_mask.shape != (batch, frames):
            raise ValueError("timestamps and frame_valid_mask must have shape [B,T]")
        if base_logits.shape != (batch, self.num_actions):
            raise ValueError("base_logits must have shape [B,A]")
        if frames < 2:
            raise ValueError("at least two history frames are required")

        pair_valid = frame_valid_mask[:, 1:] & frame_valid_mask[:, :-1]
        dt = (timestamps[:, 1:] - timestamps[:, :-1]).abs().clamp_min(1e-3)
        velocity = (history_action_tokens[:, 1:] - history_action_tokens[:, :-1]) / dt[:, :, None, None]
        acceleration = torch.zeros_like(velocity)
        if frames > 2:
            midpoint_dt = 0.5 * (dt[:, 1:] + dt[:, :-1]).clamp_min(1e-3)
            acceleration[:, 1:] = (velocity[:, 1:] - velocity[:, :-1]) / midpoint_dt[:, :, None, None]

        motion = self.motion_projection(torch.cat((velocity, acceleration), dim=-1))
        patch_displacement = action_nodes.new_zeros(batch, frames - 1, self.num_actions, 2)
        patch_confidence = action_nodes.new_zeros(batch, frames - 1, self.num_actions)
        patch_motion_energy = action_nodes.new_zeros(batch, frames - 1, self.num_actions)
        if patch_tokens is not None or patch_xy is not None or patch_weight is not None:
            if patch_tokens is None or patch_xy is None or patch_weight is None:
                raise ValueError("patch_tokens, patch_xy, and patch_weight must be provided together")
            if patch_tokens.ndim != 5 or patch_tokens.shape[:3] != (batch, frames, self.num_actions):
                raise ValueError("patch_tokens must be [B,T,A,K,D]")
            if patch_tokens.shape[-1] != self.dim or patch_xy.shape != (*patch_tokens.shape[:-1], 2):
                raise ValueError("patch token dimensions or coordinates do not match")
            if patch_weight.shape != patch_tokens.shape[:-1]:
                raise ValueError("patch_weight must be [B,T,A,K]")
            previous = torch.nn.functional.normalize(patch_tokens[:, :-1], dim=-1)
            current = torch.nn.functional.normalize(patch_tokens[:, 1:], dim=-1)
            similarity = torch.einsum("bfaid,bfajd->bfaij", previous, current) / 0.07
            correspondence = torch.softmax(similarity, dim=-1)
            matched_xy = torch.einsum("bfaij,bfajc->bfaic", correspondence, patch_xy[:, 1:])
            matched_token = torch.einsum("bfaij,bfajd->bfaid", correspondence, patch_tokens[:, 1:])
            displacement = matched_xy - patch_xy[:, :-1]
            source_weight = patch_weight[:, :-1]
            source_weight = source_weight / source_weight.sum(-1, keepdim=True).clamp_min(1e-8)
            patch_displacement = torch.einsum("bfak,bfakc->bfac", source_weight, displacement)
            appearance = torch.einsum(
                "bfak,bfakd->bfad", source_weight, matched_token - patch_tokens[:, :-1]
            )
            magnitude = displacement.square().sum(-1).sqrt()
            radial = (displacement * patch_xy[:, :-1]).sum(-1)
            patch_motion_energy = (source_weight * magnitude).sum(-1)
            expansion = (source_weight * radial).sum(-1)
            patch_confidence = (source_weight * correspondence.max(-1).values).sum(-1)
            patch_descriptor = torch.cat(
                (
                    appearance,
                    patch_displacement,
                    patch_motion_energy[..., None],
                    expansion[..., None],
                    patch_confidence[..., None],
                ),
                dim=-1,
            )
            motion = motion + self.patch_motion_projection(patch_descriptor)
        motion = motion + self.action_identity[None, None]
        motion_flat = motion.flatten(1, 2)
        key = self.key_projection(motion_flat)
        value = self.value_projection(motion_flat)

        confidence = torch.sigmoid(base_logits.detach())
        uncertainty = 1.0 - (2.0 * confidence - 1.0).abs()
        query = action_nodes + self.action_identity[None]
        query = query + self.logit_context(torch.stack((base_logits.detach(), uncertainty), dim=-1))
        query = self.query_projection(query)
        scores = torch.einsum("bad,bnd->ban", query, key) / math.sqrt(self.dim)

        source_action = torch.arange(self.num_actions, device=scores.device).repeat(frames - 1)
        target_action = torch.arange(self.num_actions, device=scores.device)[:, None]
        same_action = source_action[None] == target_action
        scores = scores + self.same_action_bias * same_action[None].to(scores.dtype)
        flat_valid = pair_valid[:, :, None].expand(-1, -1, self.num_actions).flatten(1)
        attention = self._safe_attention(scores, flat_valid[:, None].expand(-1, self.num_actions, -1))
        context = torch.einsum("ban,bnd->bad", attention, value)
        hidden = self.readout(torch.cat((action_nodes, context), dim=-1))
        history_available = pair_valid.any(-1)
        delta = self.cap * torch.tanh(self.output(hidden).squeeze(-1)) * torch.sigmoid(self.gate(hidden).squeeze(-1))
        delta = delta * history_available[:, None].to(delta.dtype)

        same_action_mass = (attention * same_action[None].to(attention.dtype)).sum(-1)
        motion_energy = velocity.square().mean(dim=(-1, -2)).sqrt()
        motion_energy = motion_energy * pair_valid.to(motion_energy.dtype)
        return {
            "traffic_action_delta": delta,
            "traffic_action_context": context,
            "traffic_action_attention": attention,
            "traffic_same_action_mass": same_action_mass,
            "traffic_motion_energy": motion_energy,
            "traffic_patch_displacement": patch_displacement,
            "traffic_patch_match_confidence": patch_confidence,
            "traffic_patch_motion_energy": patch_motion_energy,
            "traffic_history_available": history_available,
        }
