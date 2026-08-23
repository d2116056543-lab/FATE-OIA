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
            "traffic_history_available": history_available,
        }
