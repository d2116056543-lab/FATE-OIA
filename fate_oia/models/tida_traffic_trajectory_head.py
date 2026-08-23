from __future__ import annotations

import math

import torch
from torch import nn

from .acpr_sparse_ops import entmax15_bisect


class TIDATrafficTrajectoryHead(nn.Module):
    """Convert identity-consistent traffic trajectories into an action-only residual."""

    def __init__(
        self,
        dim: int = 384,
        num_actions: int = 4,
        num_heads: int = 4,
        cap: float = 0.08,
    ) -> None:
        super().__init__()
        if dim <= 0 or num_actions <= 0 or num_heads <= 0 or dim % num_heads or cap <= 0:
            raise ValueError("invalid trajectory head dimensions")
        self.dim = int(dim)
        self.num_actions = int(num_actions)
        self.cap = float(cap)
        self.action_identity = nn.Parameter(torch.randn(num_actions, dim) * 0.02)
        self.motion_projection = nn.Sequential(nn.Linear(9, dim), nn.GELU(), nn.LayerNorm(dim))
        self.direction_projection = nn.Sequential(nn.Linear(8, dim), nn.GELU())
        temporal_layer = nn.TransformerEncoderLayer(
            dim, num_heads, 2 * dim, dropout=0.0, batch_first=True, norm_first=True
        )
        relation_layer = nn.TransformerEncoderLayer(
            dim, num_heads, 2 * dim, dropout=0.0, batch_first=True, norm_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, 1)
        self.relation_encoder = nn.TransformerEncoder(relation_layer, 1)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.readout = nn.Sequential(nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU())
        self.output = nn.Linear(dim, 1)
        self.trust_raw = nn.Parameter(torch.full((num_actions,), -2.0))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        angles = torch.arange(8, dtype=torch.float32) * (2.0 * math.pi / 8.0)
        self.register_buffer("direction_bins", torch.stack((angles.cos(), angles.sin()), dim=-1))

    def forward(
        self,
        action_nodes: torch.Tensor,
        trajectory_appearance: torch.Tensor,
        trajectory_xy: torch.Tensor,
        trajectory_visibility: torch.Tensor,
        trajectory_pair_valid: torch.Tensor,
        common_displacement: torch.Tensor,
        exclusive_displacement: torch.Tensor,
        anchor_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if trajectory_appearance.ndim != 5:
            raise ValueError("trajectory_appearance must be [B,A,K,T,D]")
        batch, actions, tracks, frames, dim = trajectory_appearance.shape
        if action_nodes.shape != (batch, actions, dim) or actions != self.num_actions or dim != self.dim:
            raise ValueError("action_nodes do not match trajectory dimensions")
        if trajectory_xy.shape != (batch, actions, tracks, frames, 2):
            raise ValueError("trajectory_xy shape mismatch")
        if trajectory_visibility.shape != (batch, actions, tracks, frames):
            raise ValueError("trajectory_visibility shape mismatch")
        if trajectory_pair_valid.shape != (batch, actions, tracks, frames - 1):
            raise ValueError("trajectory_pair_valid shape mismatch")
        if common_displacement.shape != (batch, frames - 1, 2):
            raise ValueError("common_displacement shape mismatch")
        if exclusive_displacement.shape != (batch, actions, tracks, frames - 1, 2):
            raise ValueError("exclusive_displacement shape mismatch")
        if anchor_weight.shape != (batch, actions, tracks):
            raise ValueError("anchor_weight must be [B,A,K]")

        displacement = trajectory_xy[..., 1:, :] - trajectory_xy[..., :-1, :]
        pair_weight = trajectory_pair_valid.to(displacement.dtype)
        speed = displacement.square().sum(-1).sqrt()
        acceleration = torch.zeros_like(displacement)
        if frames > 2:
            acceleration[..., 1:, :] = displacement[..., 1:, :] - displacement[..., :-1, :]
        acceleration_norm = acceleration.square().sum(-1).sqrt()
        radial = (displacement * trajectory_xy[..., :-1, :]).sum(-1)
        confidence = trajectory_visibility[..., 1:]
        common = common_displacement[:, None, None].expand(-1, actions, tracks, -1, -1)
        motion_features = torch.cat(
            (
                displacement,
                exclusive_displacement,
                common,
                speed[..., None],
                acceleration_norm[..., None],
                radial[..., None],
            ),
            dim=-1,
        )
        first = torch.zeros_like(motion_features[..., :1, :])
        motion_features = torch.cat((first, motion_features), dim=3)
        position = torch.arange(frames, device=trajectory_appearance.device, dtype=trajectory_appearance.dtype)
        frequency = torch.exp(
            torch.arange(0, dim, 2, device=position.device, dtype=position.dtype)
            * (-math.log(10000.0) / max(dim, 1))
        )
        temporal_position = trajectory_appearance.new_zeros(frames, dim)
        temporal_position[:, 0::2] = torch.sin(position[:, None] * frequency[None])
        temporal_position[:, 1::2] = torch.cos(position[:, None] * frequency[None, : temporal_position[:, 1::2].shape[1]])
        frame_tokens = (
            trajectory_appearance
            + self.motion_projection(motion_features)
            + 0.1 * temporal_position[None, None, None]
        )

        valid_frame = trajectory_visibility > 0
        encoded = self.temporal_encoder(
            frame_tokens.reshape(batch * actions * tracks, frames, dim),
            src_key_padding_mask=(~valid_frame).reshape(batch * actions * tracks, frames),
        ).reshape(batch, actions, tracks, frames, dim)
        recency = torch.linspace(
            0.25, 1.0, frames, device=trajectory_visibility.device, dtype=trajectory_visibility.dtype
        )
        frame_weight = trajectory_visibility * recency
        frame_weight = frame_weight / frame_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("bakt,baktd->bakd", frame_weight, encoded)
        # The terminal token is the real target-frame anchor; mixing it with a
        # recency-weighted history summary preserves identity and chronology.
        trajectory_tokens = 0.5 * (pooled + encoded[..., -1, :])

        unit = displacement / speed[..., None].clamp_min(1e-6)
        orientation = torch.einsum("baktc,hc->bakth", unit, self.direction_bins)
        soft_bins = torch.softmax(4.0 * orientation, dim=-1) * speed[..., None] * pair_weight[..., None]
        direction_histogram = soft_bins.sum(3)
        direction_histogram = direction_histogram / direction_histogram.sum(-1, keepdim=True).clamp_min(1e-8)
        trajectory_tokens = trajectory_tokens + self.direction_projection(direction_histogram)

        relation = self.relation_encoder(trajectory_tokens.reshape(batch * actions, tracks, dim))
        relation = relation.reshape(batch, actions, tracks, dim)
        query = self.query(action_nodes + self.action_identity[None])
        scores = torch.einsum("bad,bakd->bak", query, self.key(relation)) / math.sqrt(dim)
        anchor_prior = anchor_weight.detach().clamp_min(1e-8)
        anchor_prior = anchor_prior / anchor_prior.sum(-1, keepdim=True).clamp_min(1e-8)
        scores = scores + anchor_prior.log()
        track_valid = valid_frame.any(-1) & (anchor_weight > 0)
        scores = scores.masked_fill(~track_valid, -1e4)
        has_track = track_valid.any(-1, keepdim=True)
        scores = torch.where(has_track, scores, torch.zeros_like(scores))
        attention = entmax15_bisect(scores, dim=-1)
        attention = torch.where(has_track, attention, torch.zeros_like(attention))
        context = torch.einsum("bak,bakd->bad", attention, self.value(relation))
        track_support = (confidence * pair_weight).sum(-1) / pair_weight.sum(-1).clamp_min(1.0)
        trajectory_support = (attention * track_support).sum(-1)
        hidden = self.readout(torch.cat((action_nodes, context), dim=-1))
        null_hidden = self.readout(torch.cat((action_nodes, torch.zeros_like(context)), dim=-1))
        trust = torch.sigmoid(self.trust_raw)[None].expand(batch, -1)
        evidence_logit = self.output(hidden).squeeze(-1) - self.output(null_hidden).squeeze(-1)
        delta = self.cap * torch.tanh(evidence_logit) * trust * trajectory_support

        return {
            "traffic_trajectory_delta": delta,
            "traffic_trajectory_context": context,
            "traffic_trajectory_trust": trust,
            "traffic_trajectory_support": trajectory_support,
            "trajectory_attention": attention,
            "trajectory_tokens": relation,
            "trajectory_direction_histogram": direction_histogram,
            "trajectory_speed": speed * pair_weight,
            "trajectory_acceleration": acceleration_norm * pair_weight,
            "trajectory_radial_motion": radial * pair_weight,
            "trajectory_pair_confidence": confidence * pair_weight,
        }
