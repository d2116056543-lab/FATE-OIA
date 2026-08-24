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
        state_enabled: bool = True,
        state_strength_scale: float = 8.0,
        state_cap_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        if (
            dim <= 0 or num_actions <= 0 or num_heads <= 0 or dim % num_heads or cap <= 0
            or state_strength_scale <= 0 or not 0 <= state_cap_ratio <= 1
        ):
            raise ValueError("invalid trajectory head dimensions")
        self.dim = int(dim)
        self.num_actions = int(num_actions)
        self.cap = float(cap)
        self.state_enabled = bool(state_enabled)
        self.state_strength_scale = float(state_strength_scale)
        self.state_cap_ratio = float(state_cap_ratio)
        self.action_identity = nn.Parameter(torch.randn(num_actions, dim) * 0.02)
        self.motion_projection = nn.Sequential(nn.Linear(9, dim), nn.GELU(), nn.LayerNorm(dim))
        self.direction_projection = nn.Sequential(nn.Linear(8, dim), nn.GELU())
        self.interaction_projection = nn.Sequential(
            nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        nn.init.zeros_(self.interaction_projection[-1].weight)
        nn.init.zeros_(self.interaction_projection[-1].bias)
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
        self.readout = nn.Sequential(nn.LayerNorm(4 * dim), nn.Linear(4 * dim, dim), nn.GELU())
        self.state_projection = nn.Sequential(
            nn.LayerNorm(8), nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU()
        )
        self.order_gate_projection = nn.Linear(2 * dim, 1)
        self.output = nn.Linear(dim, 1)
        self.state_output = nn.Linear(dim, 1)
        self.utility_projection = nn.Linear(dim, 1)
        self.state_utility_projection = nn.Linear(dim, 1)
        self.trust_raw = nn.Parameter(torch.zeros(num_actions))
        nn.init.zeros_(self.order_gate_projection.weight)
        nn.init.zeros_(self.order_gate_projection.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        nn.init.zeros_(self.state_output.weight)
        nn.init.zeros_(self.state_output.bias)
        nn.init.zeros_(self.utility_projection.weight)
        nn.init.constant_(self.utility_projection.bias, math.log(999.0))
        nn.init.zeros_(self.state_utility_projection.weight)
        nn.init.zeros_(self.state_utility_projection.bias)
        if not self.state_enabled:
            for module in (
                self.state_projection, self.state_output, self.state_utility_projection
            ):
                for parameter in module.parameters():
                    parameter.requires_grad = False
        angles = torch.arange(8, dtype=torch.float32) * (2.0 * math.pi / 8.0)
        self.register_buffer("direction_bins", torch.stack((angles.cos(), angles.sin()), dim=-1))

    def _encode_trajectory(
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
        batch, actions, tracks, frames, dim = trajectory_appearance.shape
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
        position = torch.arange(
            frames, device=trajectory_appearance.device, dtype=trajectory_appearance.dtype
        )
        frequency = torch.exp(
            torch.arange(0, dim, 2, device=position.device, dtype=position.dtype)
            * (-math.log(10000.0) / max(dim, 1))
        )
        temporal_position = trajectory_appearance.new_zeros(frames, dim)
        temporal_position[:, 0::2] = torch.sin(position[:, None] * frequency[None])
        temporal_position[:, 1::2] = torch.cos(
            position[:, None] * frequency[None, : temporal_position[:, 1::2].shape[1]]
        )
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
            0.25, 1.0, frames,
            device=trajectory_visibility.device, dtype=trajectory_visibility.dtype,
        )
        frame_weight = trajectory_visibility * recency
        frame_weight = frame_weight / frame_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("bakt,baktd->bakd", frame_weight, encoded)
        trajectory_tokens = 0.5 * (pooled + encoded[..., -1, :])

        unit = displacement / speed[..., None].clamp_min(1e-6)
        orientation = torch.einsum("baktc,hc->bakth", unit, self.direction_bins)
        soft_bins = torch.softmax(4.0 * orientation, dim=-1) * speed[..., None] * pair_weight[..., None]
        direction_histogram = soft_bins.sum(3)
        direction_histogram = direction_histogram / direction_histogram.sum(
            -1, keepdim=True
        ).clamp_min(1e-8)
        trajectory_tokens = trajectory_tokens + self.direction_projection(direction_histogram)

        # Explicit inter-trajectory geometry complements self-attention with the
        # driving-relevant relations it otherwise has to rediscover from few clips.
        mean_velocity = (displacement * pair_weight[..., None]).sum(-2) / pair_weight.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        final_position = trajectory_xy[..., -1, :]
        relative_position = (
            final_position[:, :, None, :, :] - final_position[:, :, :, None, :]
        )
        relative_velocity = (
            mean_velocity[:, :, None, :, :] - mean_velocity[:, :, :, None, :]
        )
        distance = relative_position.square().sum(-1).sqrt()
        closing = -(relative_position * relative_velocity).sum(-1) / distance.clamp_min(1e-4)
        crossing = (
            relative_position[..., 0] * relative_velocity[..., 1]
            - relative_position[..., 1] * relative_velocity[..., 0]
        ).abs() / distance.clamp_min(1e-4)
        proximity = torch.exp(-2.0 * distance)
        closing_positive = closing.relu()
        ttc_risk = closing_positive / (distance + closing_positive + 1e-4)
        interaction_features = torch.cat(
            (
                relative_position,
                relative_velocity,
                closing[..., None],
                crossing[..., None],
                proximity[..., None],
                ttc_risk[..., None],
            ),
            dim=-1,
        )
        track_valid_for_relation = valid_frame.any(-1)
        off_diagonal = ~torch.eye(tracks, dtype=torch.bool, device=distance.device)[None, None]
        relation_valid = (
            track_valid_for_relation[..., :, None]
            & track_valid_for_relation[..., None, :]
            & off_diagonal
        )
        relation_weight = proximity * relation_valid.to(proximity.dtype)
        relation_weight = relation_weight / relation_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        interaction_summary = torch.einsum(
            "bakj,bakjf->bakf", relation_weight, interaction_features
        )
        interaction_risk = torch.einsum(
            "bakj,bakj->bak", relation_weight, 0.5 * (ttc_risk + crossing.tanh())
        )
        trajectory_tokens = trajectory_tokens + self.interaction_projection(interaction_summary)

        # Geometry-only traffic state survives time-arrow differencing without
        # opening an appearance shortcut into the action decision. Each action
        # receives the exclusive motion and interaction state of its own tracks.
        exclusive_speed = exclusive_displacement.square().sum(-1).sqrt()
        exclusive_acceleration = torch.zeros_like(exclusive_speed)
        if frames > 2:
            exclusive_acceleration[..., 1:] = (
                exclusive_displacement[..., 1:, :] - exclusive_displacement[..., :-1, :]
            ).square().sum(-1).sqrt()
        state_pair_weight = pair_weight * confidence
        state_denominator = state_pair_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        mean_exclusive = (
            exclusive_displacement * state_pair_weight[..., None]
        ).sum(-2) / state_denominator
        mean_abs_exclusive = (
            exclusive_displacement.abs() * state_pair_weight[..., None]
        ).sum(-2) / state_denominator
        mean_exclusive_speed = (
            exclusive_speed * state_pair_weight
        ).sum(-1, keepdim=True) / state_denominator
        max_exclusive_speed = (
            exclusive_speed.masked_fill(state_pair_weight <= 0, 0.0).amax(-1, keepdim=True)
        )
        mean_exclusive_acceleration = (
            exclusive_acceleration * state_pair_weight
        ).sum(-1, keepdim=True) / state_denominator
        state_track_features = torch.cat(
            (
                mean_exclusive,
                mean_abs_exclusive,
                mean_exclusive_speed,
                max_exclusive_speed,
                mean_exclusive_acceleration,
                interaction_risk[..., None],
            ),
            dim=-1,
        )
        state_track_support = state_pair_weight.mean(-1)
        state_track_weight = anchor_weight.detach().clamp_min(0.0) * state_track_support
        state_track_weight = state_track_weight / state_track_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        motion_state_features = torch.einsum("bak,bakf->baf", state_track_weight, state_track_features)

        relation = self.relation_encoder(
            trajectory_tokens.reshape(batch * actions, tracks, dim)
        ).reshape(batch, actions, tracks, dim)
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
        return {
            "context": context,
            "support": (attention * track_support).sum(-1),
            "attention": attention,
            "tokens": relation,
            "direction_histogram": direction_histogram,
            "speed": speed * pair_weight,
            "acceleration": acceleration_norm * pair_weight,
            "radial": radial * pair_weight,
            "pair_confidence": confidence * pair_weight,
            "interaction_risk": interaction_risk,
            "interaction_summary": interaction_summary,
            "motion_state_features": motion_state_features,
        }

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
        base_action_logits: torch.Tensor | None = None,
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
        if base_action_logits is None:
            base_action_logits = action_nodes.new_zeros(batch, actions)
        if base_action_logits.shape != (batch, actions):
            raise ValueError("base_action_logits must be [B,A]")

        ordered = self._encode_trajectory(
            action_nodes, trajectory_appearance, trajectory_xy, trajectory_visibility,
            trajectory_pair_valid, common_displacement, exclusive_displacement, anchor_weight,
        )
        reversed_order = self._encode_trajectory(
            action_nodes,
            trajectory_appearance.flip(3),
            trajectory_xy.flip(3),
            trajectory_visibility.flip(3),
            trajectory_pair_valid.flip(3),
            -common_displacement.flip(1),
            -exclusive_displacement.flip(3),
            anchor_weight,
        )
        order_contrast = ordered["context"] - reversed_order["context"]
        action_identity = self.action_identity[None].expand(batch, -1, -1)

        def credit_features(contrast: torch.Tensor) -> torch.Tensor:
            magnitude = contrast.abs()
            return torch.cat(
                (contrast, magnitude, action_identity * contrast, action_identity * magnitude), dim=-1
            )

        hidden = self.readout(credit_features(order_contrast))
        reverse_hidden = self.readout(credit_features(-order_contrast))
        state_hidden = self.state_projection(ordered["motion_state_features"])
        trust = torch.sigmoid(self.trust_raw)[None].expand(batch, -1)
        ordered_readout = self.output(hidden).squeeze(-1)
        reversed_readout = self.output(reverse_hidden).squeeze(-1)
        # Temporal credit must exclude any order-independent class prior or output bias.
        evidence_logit = 0.5 * (ordered_readout - reversed_readout)
        control_logit = -evidence_logit
        order_rms = order_contrast.square().mean(-1).sqrt()
        learned_order_gate = torch.sigmoid(
            self.order_gate_projection(
                torch.cat((order_contrast.abs(), action_identity * order_contrast.abs()), dim=-1)
            ).squeeze(-1)
        )
        order_strength = 1.0 - torch.exp(-order_rms)
        order_gate = order_strength * (0.5 + 0.5 * learned_order_gate)
        uncertainty_gate = 0.25 + 0.75 * torch.exp(-base_action_logits.detach().abs())
        trajectory_support = 0.5 * (ordered["support"] + reversed_order["support"])
        support_gate = trajectory_support / (trajectory_support + 0.05)
        order_budget = self.cap * trust * support_gate * order_gate * uncertainty_gate
        order_delta = order_budget * torch.tanh(evidence_logit)
        order_control_delta = order_budget * torch.tanh(control_logit)
        state_logit = self.state_output(state_hidden).squeeze(-1)
        state_strength = 1.0 - torch.exp(
            -self.state_strength_scale * ordered["motion_state_features"].abs().mean(-1)
        )
        state_budget = (
            self.cap * self.state_cap_ratio * trust * support_gate * uncertainty_gate * state_strength
        )
        state_delta = state_budget * torch.tanh(state_logit)
        if not self.state_enabled:
            state_logit = torch.zeros_like(state_logit)
            state_strength = torch.zeros_like(state_strength)
            state_delta = torch.zeros_like(state_delta)
        candidate_delta = (order_delta + state_delta).clamp(-self.cap, self.cap)
        candidate_control_delta = order_control_delta.clamp(-self.cap, self.cap)
        utility_logit = self.utility_projection(hidden).squeeze(-1)
        if self.state_enabled:
            utility_logit = utility_logit + self.state_utility_projection(state_hidden).squeeze(-1)
        utility_gate = torch.sigmoid(utility_logit)
        delta = utility_gate * candidate_delta
        control_delta = utility_gate * candidate_control_delta

        return {
            "traffic_trajectory_delta": delta,
            "traffic_trajectory_control_delta": control_delta,
            "traffic_trajectory_credit_logit": evidence_logit,
            "traffic_trajectory_control_logit": control_logit,
            "traffic_trajectory_candidate_delta": candidate_delta,
            "traffic_trajectory_order_delta": order_delta,
            "traffic_trajectory_state_delta": state_delta,
            "traffic_trajectory_state_effective_delta": utility_gate * state_delta,
            "traffic_trajectory_state_logit": state_logit,
            "traffic_trajectory_state_features": ordered["motion_state_features"],
            "trajectory_state_strength": state_strength,
            "traffic_trajectory_utility_logit": utility_logit,
            "traffic_trajectory_utility_gate": utility_gate,
            "traffic_trajectory_context": ordered["context"],
            "traffic_trajectory_trust": trust,
            "traffic_trajectory_support": trajectory_support,
            "trajectory_support_gate": support_gate,
            "trajectory_order_gate": order_gate,
            "trajectory_uncertainty_gate": uncertainty_gate,
            "trajectory_attention": ordered["attention"],
            "trajectory_tokens": ordered["tokens"],
            "trajectory_direction_histogram": ordered["direction_histogram"],
            "trajectory_speed": ordered["speed"],
            "trajectory_acceleration": ordered["acceleration"],
            "trajectory_radial_motion": ordered["radial"],
            "trajectory_pair_confidence": ordered["pair_confidence"],
            "trajectory_interaction_risk": ordered["interaction_risk"],
            "trajectory_interaction_summary": ordered["interaction_summary"],
            "trajectory_order_contrast_rms": order_rms,
        }
