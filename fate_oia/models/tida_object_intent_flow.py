from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .acpr_sparse_ops import entmax15_bisect


class TIDAObjectRoleHead(nn.Module):
    """Predict object/road roles from terminal DINO patch evidence only."""

    def __init__(self, dim: int = 384, num_roles: int = 5) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(),
            nn.Linear(dim // 2, num_roles),
        )

    def forward(self, patch_or_track_tokens: torch.Tensor) -> torch.Tensor:
        if patch_or_track_tokens.ndim != 3:
            raise ValueError("role head input must be [B,N,D]")
        return self.net(patch_or_track_tokens)


class _PrivateObjectIntentEncoder(nn.Module):
    def __init__(self, dim: int, motion_dim: int, num_roles: int) -> None:
        super().__init__()
        self.semantic = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.role_semantic = nn.Linear(num_roles, dim, bias=False)
        self.motion = nn.Sequential(nn.Linear(motion_dim + num_roles, dim), nn.GELU(), nn.LayerNorm(dim))
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.motion_query = nn.Linear(dim, dim, bias=False)
        self.motion_key = nn.Linear(dim, dim, bias=False)
        self.motion_value = nn.Linear(dim, dim, bias=False)
        self.interaction = nn.Linear(dim * 3, dim, bias=False)
        self.motion_mix = nn.Linear(dim * 3, 1)

    @staticmethod
    def _sparse_route(
        query: torch.Tensor,
        key: torch.Tensor,
        support: torch.Tensor,
        valid_track: torch.Tensor,
        track_mask: torch.Tensor | None,
        score_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        score = torch.einsum("bld,bkd->blk", query, key) / math.sqrt(key.shape[-1])
        score = score + support.clamp_min(1e-8).log()[:, None]
        if score_bias is not None:
            if score_bias.shape == support.shape:
                score_bias = score_bias[:, None]
            if score_bias.shape not in (score.shape, (score.shape[0], 1, score.shape[2])):
                raise ValueError("score_bias must be [B,K] or broadcastable to [B,L,K]")
            score = score + score_bias
        valid = valid_track[:, None].expand_as(score)
        if track_mask is not None:
            if track_mask.shape == valid_track.shape:
                track_mask = track_mask[:, None].expand_as(score)
            if track_mask.shape != score.shape:
                raise ValueError("track_mask must be [B,K] or [B,L,K]")
            valid = valid & track_mask
        score = score.masked_fill(~valid, -1e4)
        attention = entmax15_bisect(score, dim=-1) * valid.to(score.dtype)
        return attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)

    def forward(
        self,
        target_nodes: torch.Tensor,
        semantics: torch.Tensor,
        motion: torch.Tensor,
        support: torch.Tensor,
        valid_track: torch.Tensor,
        interaction_risk: torch.Tensor,
        role_probs: torch.Tensor,
        track_mask: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        semantic_track = self.semantic(semantics) + self.role_semantic(role_probs)
        motion_track = self.motion(torch.cat((motion, role_probs), dim=-1))
        semantic_attention = self._sparse_route(
            self.query(target_nodes), self.key(semantic_track),
            support, valid_track, track_mask,
        )
        motion_attention = self._sparse_route(
            self.motion_query(target_nodes), self.motion_key(motion_track),
            support, valid_track, track_mask,
        )
        semantic_evidence = torch.einsum(
            "blk,bkd->bld", semantic_attention, self.value(semantic_track)
        )
        motion_evidence = torch.einsum(
            "blk,bkd->bld", motion_attention, self.motion_value(motion_track)
        )
        interaction_input = torch.cat(
            (semantic_evidence, motion_evidence, semantic_evidence * motion_evidence),
            dim=-1,
        )
        motion_mix = 0.20 + 0.80 * torch.sigmoid(
            self.motion_mix(interaction_input)
        ).squeeze(-1)
        evidence = semantic_evidence + motion_mix[..., None] * self.interaction(
            interaction_input
        )
        joint_attention = (
            (1.0 - motion_mix[..., None]) * semantic_attention
            + motion_mix[..., None] * motion_attention
        )
        transported_support = torch.einsum("blk,bk->bl", joint_attention, support)
        return (
            evidence,
            joint_attention,
            transported_support,
            semantic_attention,
            motion_attention,
            motion_mix,
        )


class _PrivatePairInteractionEncoder(nn.Module):
    """Route each target over ordered future-interaction pairs."""

    def __init__(self, dim: int, pair_dim: int, num_roles: int) -> None:
        super().__init__()
        self.semantic = nn.Sequential(
            nn.LayerNorm(dim * 4), nn.Linear(dim * 4, dim), nn.GELU(),
        )
        self.motion = nn.Sequential(
            nn.Linear(pair_dim + 2 * num_roles, dim), nn.GELU(), nn.LayerNorm(dim),
        )
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        target_nodes: torch.Tensor,
        semantics: torch.Tensor,
        pair_features: torch.Tensor,
        pair_support: torch.Tensor,
        pair_valid: torch.Tensor,
        role_probs: torch.Tensor,
        pair_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, tracks, dim = semantics.shape
        source = torch.div(pair_indices, tracks, rounding_mode="floor")
        target = pair_indices.remainder(tracks)
        left = torch.gather(
            semantics, 1, source[..., None].expand(-1, -1, dim)
        )
        right = torch.gather(
            semantics, 1, target[..., None].expand(-1, -1, dim)
        )
        semantic_pair = self.semantic(torch.cat(
            (left, right, (left - right).abs(), left * right), dim=-1,
        ))
        left_role = torch.gather(
            role_probs, 1,
            source[..., None].expand(-1, -1, role_probs.shape[-1]),
        )
        right_role = torch.gather(
            role_probs, 1,
            target[..., None].expand(-1, -1, role_probs.shape[-1]),
        )
        selected_features = torch.gather(
            pair_features.flatten(1, 2), 1,
            pair_indices[..., None].expand(-1, -1, pair_features.shape[-1]),
        )
        pair_token = semantic_pair + self.motion(torch.cat(
            (selected_features, left_role, right_role), dim=-1,
        ))
        flat_support_all = pair_support.reshape(batch, tracks * tracks)
        flat_support = torch.gather(flat_support_all, 1, pair_indices)
        if pair_valid.ndim == 3:
            selected_valid = torch.gather(
                pair_valid.reshape(batch, tracks * tracks), 1, pair_indices
            )
            flat_valid = selected_valid[:, None].expand(
                -1, target_nodes.shape[1], -1
            )
        elif pair_valid.ndim == 4 and pair_valid.shape[1] == target_nodes.shape[1]:
            flat_valid = torch.gather(
                pair_valid.reshape(batch, target_nodes.shape[1], tracks * tracks),
                2,
                pair_indices[:, None].expand(-1, target_nodes.shape[1], -1),
            )
        else:
            raise ValueError("pair_valid must be [B,K,K] or [B,L,K,K]")
        score = torch.einsum(
            "bld,bpd->blp", self.query(target_nodes), self.key(pair_token)
        ) / math.sqrt(dim)
        score = score + flat_support.clamp_min(1e-8).log()[:, None]
        valid = flat_valid
        score = score.masked_fill(~valid, -1e4)
        attention = entmax15_bisect(score, dim=-1) * valid.to(score.dtype)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        pair_values = self.value(pair_token)
        evidence = torch.einsum("blp,bpd->bld", attention, pair_values)
        transported_support = torch.einsum("blp,bp->bl", attention, flat_support)
        return (
            evidence,
            torch.zeros(
                batch, target_nodes.shape[1], tracks * tracks,
                device=attention.device, dtype=attention.dtype,
            ).scatter(
                2, pair_indices[:, None].expand(-1, target_nodes.shape[1], -1), attention,
            ).reshape(batch, target_nodes.shape[1], tracks, tracks),
            transported_support,
            pair_values,
            pair_indices,
        )


class _PrivateUtilityHead(nn.Module):
    """Predict whether a detached traffic correction helps one target."""

    def __init__(self, dim: int, scalar_features: int = 6) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(dim * 3 + scalar_features),
            nn.Linear(dim * 3 + scalar_features, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(
        self,
        target: torch.Tensor,
        unary_evidence: torch.Tensor,
        pair_evidence: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (target, unary_evidence, pair_evidence, scalars), dim=-1
        ).detach()
        return self.network(features).squeeze(-1)


class TIDAObjectIntentTransport(nn.Module):
    """Target-private motion-semantic transport over reliable object tracks."""

    HORIZONS = (0.5, 1.0, 2.0, 3.0)
    MOTION_DIM = 22
    PAIR_DIM = 12

    def __init__(
        self,
        dim: int = 384,
        num_actions: int = 4,
        num_reasons: int = 21,
        heads: int = 4,
        action_cap: float = 0.08,
        reason_cap: float = 0.06,
        reason_traffic_indices: tuple[int, ...] | None = None,
        num_roles: int = 5,
        role_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads:
            raise ValueError("dim must be positive and divisible by heads")
        if action_cap <= 0 or reason_cap <= 0:
            raise ValueError("object intent caps must be positive")
        self.num_actions = int(num_actions)
        self.num_reasons = int(num_reasons)
        self.action_cap = float(action_cap)
        self.reason_cap = float(reason_cap)
        self.num_roles = int(num_roles)
        reason_mask = torch.ones(num_reasons, dtype=torch.bool)
        if reason_traffic_indices is not None:
            reason_mask.zero_()
            for index in reason_traffic_indices:
                if not 0 <= int(index) < num_reasons:
                    raise ValueError("reason traffic index is out of range")
                reason_mask[int(index)] = True
        self.register_buffer("reason_traffic_mask", reason_mask)
        self.register_buffer("action_deploy_gate", torch.zeros(num_actions))
        self.register_buffer("reason_deploy_gate", torch.zeros(num_reasons))
        self.register_buffer("action_deploy_scale", torch.zeros(num_actions))
        self.register_buffer("reason_deploy_scale", torch.zeros(num_reasons))
        self.register_buffer("action_utility_cutoff", torch.ones(num_actions))
        self.register_buffer("reason_utility_cutoff", torch.ones(num_reasons))

        # No tensor is shared between action and reason transport. This is the
        # task firewall: explanation gradients cannot move the action route.
        self.role_head = TIDAObjectRoleHead(dim, self.num_roles)
        if role_checkpoint:
            payload = torch.load(role_checkpoint, map_location="cpu", weights_only=True)
            state = payload.get("role_head", payload)
            self.role_head.load_state_dict(state, strict=True)
        self.freeze_role_head()
        self.action_encoder = _PrivateObjectIntentEncoder(dim, self.MOTION_DIM, self.num_roles)
        self.reason_encoder = _PrivateObjectIntentEncoder(dim, self.MOTION_DIM, self.num_roles)
        self.action_pair_encoder = _PrivatePairInteractionEncoder(
            dim, self.PAIR_DIM, self.num_roles
        )
        self.reason_pair_encoder = _PrivatePairInteractionEncoder(
            dim, self.PAIR_DIM, self.num_roles
        )
        self.action_output = nn.Linear(dim, 1, bias=False)
        self.reason_output = nn.Linear(dim, 1, bias=False)
        self.action_pair_output = nn.Linear(dim, 1, bias=False)
        self.reason_pair_output = nn.Linear(dim, 1, bias=False)
        self.action_utility = _PrivateUtilityHead(dim)
        self.reason_utility = _PrivateUtilityHead(dim)
        nn.init.zeros_(self.action_output.weight)
        nn.init.zeros_(self.reason_output.weight)
        nn.init.zeros_(self.action_pair_output.weight)
        nn.init.zeros_(self.reason_pair_output.weight)

    def freeze_role_head(self) -> None:
        self.role_head.eval()
        for parameter in self.role_head.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def set_deployment_gates(
        self,
        action_gate: torch.Tensor,
        reason_gate: torch.Tensor,
        *,
        source: str,
    ) -> None:
        provenance = str(source).lower()
        if "train_calib" not in provenance or "test" in provenance or "oracle" in provenance:
            raise ValueError("object-intent deployment gates must come from train_calib")
        action_gate = torch.as_tensor(action_gate, device=self.action_deploy_gate.device)
        reason_gate = torch.as_tensor(reason_gate, device=self.reason_deploy_gate.device)
        if action_gate.shape != self.action_deploy_gate.shape:
            raise ValueError("action deployment gate shape mismatch")
        if reason_gate.shape != self.reason_deploy_gate.shape:
            raise ValueError("reason deployment gate shape mismatch")
        if not (((action_gate == 0) | (action_gate == 1)).all()
                and ((reason_gate == 0) | (reason_gate == 1)).all()):
            raise ValueError("deployment gates must be binary")
        self.action_deploy_gate.copy_(action_gate.to(self.action_deploy_gate))
        self.reason_deploy_gate.copy_(reason_gate.to(self.reason_deploy_gate))
        self.action_deploy_scale.copy_(action_gate.to(self.action_deploy_scale))
        self.reason_deploy_scale.copy_(reason_gate.to(self.reason_deploy_scale))
        self.action_utility_cutoff.zero_()
        self.reason_utility_cutoff.zero_()

    @torch.no_grad()
    def set_deployment_policy(
        self,
        action_gate: torch.Tensor,
        reason_gate: torch.Tensor,
        *,
        action_scale: torch.Tensor,
        reason_scale: torch.Tensor,
        action_cutoff: torch.Tensor,
        reason_cutoff: torch.Tensor,
        source: str,
    ) -> None:
        provenance = str(source).lower()
        if "train_calib" not in provenance or "test" in provenance or "oracle" in provenance:
            raise ValueError("object-intent deployment policy must come from train_calib")
        values = {
            "action_gate": (action_gate, self.action_deploy_gate),
            "reason_gate": (reason_gate, self.reason_deploy_gate),
            "action_scale": (action_scale, self.action_deploy_scale),
            "reason_scale": (reason_scale, self.reason_deploy_scale),
            "action_cutoff": (action_cutoff, self.action_utility_cutoff),
            "reason_cutoff": (reason_cutoff, self.reason_utility_cutoff),
        }
        for name, (value, destination) in values.items():
            value = torch.as_tensor(value, device=destination.device, dtype=destination.dtype)
            if value.shape != destination.shape:
                raise ValueError(f"{name} shape mismatch")
            if "gate" in name and not (((value == 0) | (value == 1)).all()):
                raise ValueError("deployment gates must be binary")
            if "scale" in name and not ((value >= -64) & (value <= 64)).all():
                raise ValueError("deployment scales must be within [-64, 64]")
            if "cutoff" in name and not ((value >= 0) & (value <= 1)).all():
                raise ValueError("utility cutoffs must be probabilities")
            destination.copy_(value)

    @staticmethod
    def _apply_deployment_policy(
        candidate: torch.Tensor,
        utility_gate: torch.Tensor,
        label_gate: torch.Tensor,
        scale: torch.Tensor,
        cutoff: torch.Tensor,
        cap: float,
    ) -> torch.Tensor:
        selected = utility_gate >= cutoff[None]
        scaled = (candidate * scale[None]).clamp(-float(cap), float(cap))
        return label_gate[None] * selected.to(candidate.dtype) * scaled

    @staticmethod
    def sample_terminal_semantics(
        patch_tokens: torch.Tensor,
        grid_hw: tuple[int, int],
        terminal_xy: torch.Tensor,
    ) -> torch.Tensor:
        if patch_tokens.ndim != 3 or terminal_xy.ndim != 3 or terminal_xy.shape[-1] != 2:
            raise ValueError("patch_tokens must be [B,N,D] and terminal_xy [B,K,2]")
        height, width = grid_hw
        if height * width != patch_tokens.shape[1]:
            raise ValueError("grid_hw does not match patch count")
        field = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0], patch_tokens.shape[2], height, width
        )
        grid = terminal_xy.clamp(-1.0, 1.0)[:, :, None]
        sampled = F.grid_sample(
            field, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return sampled.squeeze(-1).transpose(1, 2)

    @classmethod
    def sample_track_aligned_semantics(
        cls,
        temporal_patch_tokens: torch.Tensor,
        grid_hw: tuple[int, int],
        trajectory_xy: torch.Tensor,
        visibility: torch.Tensor,
        *,
        terminal_patch_tokens: torch.Tensor | None = None,
        terminal_grid_hw: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample and pool visual semantics along each tracked trajectory."""
        if temporal_patch_tokens.ndim != 4:
            raise ValueError("temporal_patch_tokens must be [B,T,N,D]")
        if trajectory_xy.ndim != 4 or trajectory_xy.shape[-1] != 2:
            raise ValueError("trajectory_xy must be [B,T,K,2]")
        if temporal_patch_tokens.shape[:2] != trajectory_xy.shape[:2]:
            raise ValueError("temporal patch and trajectory time axes must match")
        if visibility.shape != trajectory_xy.shape[:-1]:
            raise ValueError("visibility must be [B,T,K]")
        batch, frames, patches, dim = temporal_patch_tokens.shape
        flat_tokens = temporal_patch_tokens.reshape(batch * frames, patches, dim)
        flat_xy = trajectory_xy.reshape(batch * frames, trajectory_xy.shape[2], 2)
        sampled = cls.sample_terminal_semantics(flat_tokens, grid_hw, flat_xy).reshape(
            batch, frames, trajectory_xy.shape[2], dim
        )
        if terminal_patch_tokens is not None:
            if terminal_grid_hw is None:
                raise ValueError("terminal_grid_hw is required with terminal_patch_tokens")
            sampled = sampled.clone()
            sampled[:, -1] = cls.sample_terminal_semantics(
                terminal_patch_tokens, terminal_grid_hw, trajectory_xy[:, -1]
            )

        # Every visible frame contributes, while recent observations carry more
        # identity weight. Invisible frames are exactly excluded.
        age = torch.linspace(1.0, 0.0, frames, device=sampled.device, dtype=sampled.dtype)
        recency = torch.exp(-1.5 * age)[None, :, None]
        weights = visibility.to(sampled.dtype) * recency
        no_visible = weights.sum(1) <= 0
        if no_visible.any():
            weights = weights.clone()
            weights[:, -1] = torch.where(
                no_visible, torch.ones_like(weights[:, -1]), weights[:, -1]
            )
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-8)
        pooled = torch.einsum("btk,btkd->bkd", weights, sampled)
        return sampled, pooled, weights

    @staticmethod
    def _ego_compensated_motion(
        trajectory_xy: torch.Tensor,
        visibility: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if trajectory_xy.ndim != 4 or trajectory_xy.shape[-1] != 2:
            raise ValueError("trajectory_xy must be [B,T,K,2]")
        if visibility.shape != trajectory_xy.shape[:-1]:
            raise ValueError("visibility must be [B,T,K]")
        if timestamps is None:
            timestamps = torch.arange(
                trajectory_xy.shape[1], device=trajectory_xy.device, dtype=trajectory_xy.dtype,
            )[None].expand(trajectory_xy.shape[0], -1)
        timestamps = timestamps.to(device=trajectory_xy.device, dtype=trajectory_xy.dtype)
        if timestamps.shape != trajectory_xy.shape[:2]:
            raise ValueError("timestamps must be [B,T]")
        dt = timestamps[:, 1:] - timestamps[:, :-1]
        if not torch.all(dt > 0):
            raise ValueError("timestamps must be strictly increasing")
        visible = visibility.bool()
        pair_valid = visible[:, 1:] & visible[:, :-1]
        displacement = trajectory_xy[:, 1:] - trajectory_xy[:, :-1]
        velocity = displacement / dt[:, :, None, None]
        masked = velocity.masked_fill(~pair_valid[..., None], float("nan"))
        common = torch.nanmedian(masked, dim=2).values
        common = torch.nan_to_num(common)
        exclusive_velocity = velocity - common[:, :, None]
        exclusive_velocity = exclusive_velocity * pair_valid[..., None].to(velocity.dtype)
        velocity_denominator = pair_valid.sum(1).clamp_min(1).to(velocity.dtype)[..., None]
        mean_velocity = exclusive_velocity.sum(1) / velocity_denominator
        reverse_pair = pair_valid.flip(1).to(torch.int64).argmax(1)
        last_pair = pair_valid.shape[1] - 1 - reverse_pair
        last_velocity = torch.gather(
            exclusive_velocity,
            1,
            last_pair[:, None, :, None].expand(-1, 1, -1, 2),
        ).squeeze(1)
        last_velocity = last_velocity * pair_valid.any(1)[..., None].to(last_velocity.dtype)
        acceleration = torch.zeros_like(exclusive_velocity)
        acceleration_valid = torch.zeros_like(pair_valid)
        if exclusive_velocity.shape[1] > 1:
            acceleration_dt = 0.5 * (dt[:, 1:] + dt[:, :-1])
            acceleration[:, 1:] = (
                exclusive_velocity[:, 1:] - exclusive_velocity[:, :-1]
            ) / acceleration_dt[:, :, None, None]
            acceleration_valid[:, 1:] = pair_valid[:, 1:] & pair_valid[:, :-1]
            acceleration = acceleration * acceleration_valid[..., None].to(acceleration.dtype)
        acceleration_denominator = acceleration_valid.sum(1).clamp_min(1).to(
            acceleration.dtype
        )[..., None]
        mean_acceleration = acceleration.sum(1) / acceleration_denominator
        speed = exclusive_velocity.square().sum(-1).sqrt()
        mean_speed = speed.sum(1) / velocity_denominator.squeeze(-1)
        path_length = (
            displacement.square().sum(-1).sqrt() * pair_valid.to(displacement.dtype)
        ).sum(1)
        final_xy = trajectory_xy[:, -1]
        # grid_sample coordinates place the road-facing ego anchor at the
        # bottom centre, not at the image centre. Measuring approach against
        # (0, 0) incorrectly treats motion through the horizon as ego closing.
        ego_anchor = final_xy.new_tensor((0.0, 1.0))
        ego_relative_xy = final_xy - ego_anchor
        distance = ego_relative_xy.square().sum(-1).sqrt()
        radial_velocity = (
            mean_velocity * ego_relative_xy
        ).sum(-1) / distance.clamp_min(1e-4)
        closing = (-radial_velocity).relu()
        ttc_risk = closing / (distance + closing + 1e-4)
        crossing = (
            ego_relative_xy[..., 0] * mean_velocity[..., 1]
            - ego_relative_xy[..., 1] * mean_velocity[..., 0]
        ).abs() / distance.clamp_min(1e-4)
        horizons = trajectory_xy.new_tensor(TIDAObjectIntentTransport.HORIZONS)
        future = (
            final_xy[:, :, None]
            + mean_velocity[:, :, None] * horizons[None, None, :, None]
            + 0.5 * mean_acceleration[:, :, None] * horizons[None, None, :, None].square()
        ).clamp(-1.5, 1.5)
        future_ego_distance = (future - ego_anchor).square().sum(-1).sqrt()
        future_approach_risk = (
            distance - future_ego_distance.amin(-1)
        ).relu() / distance.clamp_min(1e-4)
        # Tangential motion is only safety-relevant near the ego corridor.
        crossing_risk = crossing / (distance + crossing + 1e-4)
        crossing_risk = crossing_risk * (1.0 - distance / 2.5).clamp(0.0, 1.0)
        visible_fraction = visible.float().mean(1)
        support = visible_fraction * pair_valid.float().mean(1)
        valid_track = support > 0
        motion = torch.cat(
            (
                final_xy,
                mean_velocity,
                last_velocity,
                mean_acceleration,
                mean_speed[..., None],
                radial_velocity[..., None],
                visible_fraction[..., None],
                path_length[..., None],
                ttc_risk[..., None],
                crossing[..., None],
                future.flatten(2),
            ),
            dim=-1,
        )
        return {
            "motion": motion,
            "future_xy": future,
            "future_ego_distance": future_ego_distance,
            "future_approach_risk": future_approach_risk,
            "ego_relative_xy": ego_relative_xy,
            "support": support,
            "valid_track": valid_track,
            "common_displacement": common,
            "exclusive_displacement": exclusive_velocity * dt[:, :, None, None],
            "mean_velocity": mean_velocity,
            "mean_acceleration": mean_acceleration,
            "interaction_risk": (
                ttc_risk + future_approach_risk + crossing_risk
            ).div(3.0).clamp(0.0, 1.0),
        }

    @staticmethod
    def _pairwise_future_geometry(
        final_xy: torch.Tensor,
        mean_velocity: torch.Tensor,
        future_xy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if final_xy.ndim != 3 or final_xy.shape[-1] != 2:
            raise ValueError("final_xy must be [B,K,2]")
        if mean_velocity.shape != final_xy.shape:
            raise ValueError("mean_velocity must match final_xy")
        if future_xy.ndim != 4 or future_xy.shape[:2] != final_xy.shape[:2]:
            raise ValueError("future_xy must be [B,K,H,2]")
        relative_xy = final_xy[:, :, None] - final_xy[:, None]
        relative_velocity = mean_velocity[:, :, None] - mean_velocity[:, None]
        future_relative = future_xy[:, :, None] - future_xy[:, None]
        future_distance = future_relative.square().sum(-1).sqrt()
        current_distance = relative_xy.square().sum(-1).sqrt()
        min_future_distance = future_distance.amin(-1)
        distance_reduction = (current_distance - min_future_distance).relu()
        crossing = (
            relative_xy[..., 0] * relative_velocity[..., 1]
            - relative_xy[..., 1] * relative_velocity[..., 0]
        ).abs() / current_distance.clamp_min(1e-4)
        features = torch.cat(
            (
                relative_xy,
                relative_velocity,
                current_distance[..., None],
                future_distance,
                min_future_distance[..., None],
                distance_reduction[..., None],
                crossing[..., None],
            ),
            dim=-1,
        )
        return {
            "features": features,
            "current_distance": current_distance,
            "future_distance": future_distance,
            "min_future_distance": min_future_distance,
            "distance_reduction": distance_reduction,
        }

    @staticmethod
    def _matched_control(attention: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        selected = attention.argmax(-1)
        selected_support = torch.gather(
            support[:, None].expand(-1, attention.shape[1], -1),
            2,
            selected[..., None],
        ).squeeze(-1)
        support_distance = (support[:, None] - selected_support[..., None]).abs()
        credit = attention / attention.amax(-1, keepdim=True).clamp_min(1e-8)
        cost = support_distance + credit
        valid = support[:, None].expand_as(cost) > 0
        valid = valid.clone()
        valid.scatter_(2, selected[..., None], False)
        cost = cost.masked_fill(~valid, float("inf"))
        control = cost.argmin(-1)
        fallback = (selected + 1) % attention.shape[-1]
        return torch.where(valid.any(-1), control, fallback)

    @staticmethod
    def _matched_pair_control(
        attention: torch.Tensor,
        pair_support: torch.Tensor,
        pair_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, labels, tracks, _ = attention.shape
        flat_attention = attention.reshape(batch, labels, tracks * tracks)
        flat_support = pair_support.reshape(batch, tracks * tracks)
        flat_valid = pair_valid.reshape(batch, tracks * tracks)
        selected = flat_attention.argmax(-1)
        selected_support = torch.gather(
            flat_support[:, None].expand(-1, labels, -1), 2, selected[..., None]
        ).squeeze(-1)
        cost = (flat_support[:, None] - selected_support[..., None]).abs()
        cost = cost + flat_attention / flat_attention.amax(-1, keepdim=True).clamp_min(1e-8)
        valid = flat_valid[:, None].expand_as(cost).clone()
        valid.scatter_(2, selected[..., None], False)
        cost = cost.masked_fill(~valid, float("inf"))
        control = cost.argmin(-1)
        fallback = (selected + 1) % (tracks * tracks)
        control = torch.where(valid.any(-1), control, fallback)
        return selected, control

    @staticmethod
    def _deleted_pair_candidate(
        output: nn.Linear,
        cap: float,
        evidence: torch.Tensor,
        attention: torch.Tensor,
        transported_support: torch.Tensor,
        pair_values: torch.Tensor,
        pair_indices: torch.Tensor,
        pair_support: torch.Tensor,
        deleted_pair: torch.Tensor,
        relevance_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, labels = deleted_pair.shape
        flat_attention = attention.flatten(2)
        selected_weight = torch.gather(
            flat_attention, 2, deleted_pair[..., None]
        ).squeeze(-1)
        selected_position = (
            pair_indices[:, None] == deleted_pair[..., None]
        ).to(torch.int64).argmax(-1)
        selected_value = torch.gather(
            pair_values[:, None].expand(-1, labels, -1, -1), 2,
            selected_position[..., None, None].expand(-1, -1, 1, pair_values.shape[-1]),
        ).squeeze(2)
        flat_support = pair_support.flatten(1)
        selected_support = torch.gather(
            flat_support[:, None].expand(-1, labels, -1), 2,
            deleted_pair[..., None],
        ).squeeze(-1)
        remaining_mass = 1.0 - selected_weight
        denominator = remaining_mass.clamp_min(1e-8)
        deleted_evidence = (
            evidence - selected_weight[..., None] * selected_value
        ) / denominator[..., None]
        deleted_support = (
            transported_support - selected_weight * selected_support
        ) / denominator
        has_control = remaining_mass > 1e-6
        deleted_evidence = deleted_evidence * has_control[..., None]
        deleted_support = deleted_support.clamp(0.0, 1.0) * has_control
        candidate = 0.5 * float(cap) * deleted_support * torch.tanh(
            output(deleted_evidence).squeeze(-1)
        )
        if relevance_mask is not None:
            candidate = candidate * relevance_mask.to(candidate.dtype)[None]
        return candidate

    @staticmethod
    def _deleted_candidate(
        encoder: _PrivateObjectIntentEncoder,
        output: nn.Linear,
        cap: float,
        target_nodes: torch.Tensor,
        semantics: torch.Tensor,
        motion: torch.Tensor,
        support: torch.Tensor,
        valid_track: torch.Tensor,
        interaction_risk: torch.Tensor,
        role_probs: torch.Tensor,
        deleted_track: torch.Tensor,
        relevance_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = valid_track[:, None].expand(
            -1, target_nodes.shape[1], -1
        ).clone()
        mask.scatter_(2, deleted_track[..., None], False)
        evidence, _, transported_support, _, _, _ = encoder(
            target_nodes, semantics, motion, support, valid_track, interaction_risk,
            role_probs, mask,
        )
        candidate = float(cap) * transported_support * torch.tanh(
            output(evidence).squeeze(-1)
        )
        if relevance_mask is not None:
            candidate = candidate * relevance_mask.to(candidate.dtype)[None]
        return candidate

    def forward(
        self,
        trajectory_xy: torch.Tensor,
        visibility: torch.Tensor,
        terminal_patch_tokens: torch.Tensor,
        grid_hw: tuple[int, int],
        action_nodes: torch.Tensor,
        reason_nodes: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        temporal_patch_tokens: torch.Tensor | None = None,
        temporal_grid_hw: tuple[int, int] | None = None,
        base_action_logits: torch.Tensor | None = None,
        base_reason_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        geometry = self._ego_compensated_motion(trajectory_xy, visibility, timestamps)
        terminal_semantics = self.sample_terminal_semantics(
            terminal_patch_tokens, grid_hw, trajectory_xy[:, -1]
        )
        if temporal_patch_tokens is None:
            temporal_semantics = terminal_semantics[:, None]
            semantic_temporal_weights = terminal_semantics.new_ones(
                terminal_semantics.shape[0], 1, terminal_semantics.shape[1]
            )
        else:
            if temporal_grid_hw is None:
                raise ValueError("temporal_grid_hw is required with temporal_patch_tokens")
            temporal_semantics, _, semantic_temporal_weights = self.sample_track_aligned_semantics(
                temporal_patch_tokens,
                temporal_grid_hw,
                trajectory_xy,
                visibility,
                terminal_patch_tokens=terminal_patch_tokens,
                terminal_grid_hw=grid_hw,
            )
        # Role labels are never accepted by forward. A frozen visual role head
        # supplies train/test-identical object identity, detached from both tasks.
        frame_role_logits = self.role_head(
            temporal_semantics.flatten(0, 1)
        ).reshape(*temporal_semantics.shape[:-1], self.num_roles)
        frame_role_probs = frame_role_logits.softmax(-1).detach()
        terminal_role = frame_role_probs[:, -1]
        role_agreement = torch.einsum(
            "btkr,bkr->btk", frame_role_probs, terminal_role
        ).clamp(0.0, 1.0)
        semantic_temporal_weights = semantic_temporal_weights * (
            0.25 + 0.75 * role_agreement
        )
        semantic_temporal_weights = semantic_temporal_weights / semantic_temporal_weights.sum(
            1, keepdim=True
        ).clamp_min(1e-8)
        semantics = torch.einsum(
            "btk,btkd->bkd", semantic_temporal_weights, temporal_semantics
        )
        role_probs = torch.einsum(
            "btk,btkr->bkr", semantic_temporal_weights, frame_role_probs
        )
        role_logits = role_probs.clamp_min(1e-8).log()
        role_consistency = torch.einsum(
            "btk,btk->bk", semantic_temporal_weights, role_agreement
        )
        foreground_probability = 1.0 - role_probs[..., 0]
        role_support = geometry["support"] * (0.15 + 0.85 * foreground_probability)
        pair_geometry = self._pairwise_future_geometry(
            trajectory_xy[:, -1], geometry["mean_velocity"], geometry["future_xy"]
        )
        pair_support = (
            role_support[:, :, None] * role_support[:, None]
        ).clamp_min(0.0).sqrt()
        tracks = role_support.shape[1]
        diagonal = torch.eye(tracks, device=role_support.device, dtype=torch.bool)[None]
        pair_valid = (
            geometry["valid_track"][:, :, None]
            & geometry["valid_track"][:, None]
            & ~diagonal
        )
        pair_priority = pair_support * (
            1.0 + pair_geometry["distance_reduction"]
        ) / (1.0 + pair_geometry["min_future_distance"])
        flat_priority = pair_priority.flatten(1).masked_fill(
            ~pair_valid.flatten(1), float("-inf")
        )
        pair_budget = min(128, tracks * tracks)
        pair_indices = flat_priority.topk(pair_budget, dim=-1).indices
        candidate_mask = torch.zeros_like(flat_priority, dtype=torch.bool)
        candidate_mask.scatter_(1, pair_indices, True)
        pair_valid = pair_valid & candidate_mask.reshape_as(pair_valid)
        (
            action_evidence,
            action_attention,
            action_support,
            action_semantic_attention,
            action_motion_attention,
            action_motion_mix,
        ) = self.action_encoder(
            action_nodes,
            semantics,
            geometry["motion"],
            role_support,
            geometry["valid_track"],
            geometry["interaction_risk"],
            role_probs,
        )
        (
            reason_evidence,
            reason_attention,
            reason_support,
            reason_semantic_attention,
            reason_motion_attention,
            reason_motion_mix,
        ) = self.reason_encoder(
            reason_nodes,
            semantics,
            geometry["motion"],
            role_support,
            geometry["valid_track"],
            geometry["interaction_risk"],
            role_probs,
        )
        (
            action_pair_evidence, action_pair_attention, action_pair_support,
            action_pair_values, action_pair_indices,
        ) = self.action_pair_encoder(
            action_nodes, semantics, pair_geometry["features"], pair_support,
            pair_valid, role_probs, pair_indices,
        )
        (
            reason_pair_evidence, reason_pair_attention, reason_pair_support,
            reason_pair_values, reason_pair_indices,
        ) = self.reason_pair_encoder(
            reason_nodes, semantics, pair_geometry["features"], pair_support,
            pair_valid, role_probs, pair_indices,
        )
        action_unary_candidate = self.action_cap * action_support * torch.tanh(
            self.action_output(action_evidence).squeeze(-1)
        )
        reason_unary_candidate = self.reason_cap * reason_support * torch.tanh(
            self.reason_output(reason_evidence).squeeze(-1)
        )
        action_pair_candidate = 0.5 * self.action_cap * action_pair_support * torch.tanh(
            self.action_pair_output(action_pair_evidence).squeeze(-1)
        )
        reason_pair_candidate = 0.5 * self.reason_cap * reason_pair_support * torch.tanh(
            self.reason_pair_output(reason_pair_evidence).squeeze(-1)
        )
        action_candidate = (action_unary_candidate + action_pair_candidate).clamp(
            -self.action_cap, self.action_cap
        )
        reason_candidate = (reason_unary_candidate + reason_pair_candidate).clamp(
            -self.reason_cap, self.reason_cap
        )
        reason_candidate = reason_candidate * self.reason_traffic_mask.to(reason_candidate.dtype)[None]
        reason_pair_candidate = reason_pair_candidate * self.reason_traffic_mask.to(
            reason_pair_candidate.dtype
        )[None]
        if base_action_logits is None:
            base_action_logits = torch.zeros_like(action_candidate)
        if base_reason_logits is None:
            base_reason_logits = torch.zeros_like(reason_candidate)
        if base_action_logits.shape != action_candidate.shape:
            raise ValueError("base_action_logits must match action candidate")
        if base_reason_logits.shape != reason_candidate.shape:
            raise ValueError("base_reason_logits must match reason candidate")
        action_entropy = -(
            action_attention * action_attention.clamp_min(1e-8).log()
        ).sum(-1)
        reason_entropy = -(
            reason_attention * reason_attention.clamp_min(1e-8).log()
        ).sum(-1)
        action_pair_entropy = -(
            action_pair_attention * action_pair_attention.clamp_min(1e-8).log()
        ).sum((-1, -2))
        reason_pair_entropy = -(
            reason_pair_attention * reason_pair_attention.clamp_min(1e-8).log()
        ).sum((-1, -2))
        action_utility_logit = self.action_utility(
            action_nodes, action_evidence, action_pair_evidence,
            torch.stack((
                base_action_logits, action_candidate, action_support,
                action_pair_support, action_entropy, action_pair_entropy,
            ), dim=-1),
        )
        reason_utility_logit = self.reason_utility(
            reason_nodes, reason_evidence, reason_pair_evidence,
            torch.stack((
                base_reason_logits, reason_candidate, reason_support,
                reason_pair_support, reason_entropy, reason_pair_entropy,
            ), dim=-1),
        )
        action_utility_gate = action_utility_logit.sigmoid()
        reason_utility_gate = reason_utility_logit.sigmoid()
        action_selected_pair, action_control_pair = self._matched_pair_control(
            action_pair_attention, pair_support, pair_valid
        )
        reason_selected_pair, reason_control_pair = self._matched_pair_control(
            reason_pair_attention, pair_support, pair_valid
        )
        action_selected_pair_deleted = self._deleted_pair_candidate(
            self.action_pair_output, self.action_cap, action_pair_evidence,
            action_pair_attention, action_pair_support, action_pair_values,
            action_pair_indices, pair_support, action_selected_pair,
        )
        action_control_pair_deleted = self._deleted_pair_candidate(
            self.action_pair_output, self.action_cap, action_pair_evidence,
            action_pair_attention, action_pair_support, action_pair_values,
            action_pair_indices, pair_support, action_control_pair,
        )
        reason_selected_pair_deleted = self._deleted_pair_candidate(
            self.reason_pair_output, self.reason_cap, reason_pair_evidence,
            reason_pair_attention, reason_pair_support, reason_pair_values,
            reason_pair_indices, pair_support, reason_selected_pair, self.reason_traffic_mask,
        )
        reason_control_pair_deleted = self._deleted_pair_candidate(
            self.reason_pair_output, self.reason_cap, reason_pair_evidence,
            reason_pair_attention, reason_pair_support, reason_pair_values,
            reason_pair_indices, pair_support, reason_control_pair, self.reason_traffic_mask,
        )
        action_selected_pair_deleted = (
            action_unary_candidate + action_selected_pair_deleted
        ).clamp(-self.action_cap, self.action_cap)
        action_control_pair_deleted = (
            action_unary_candidate + action_control_pair_deleted
        ).clamp(-self.action_cap, self.action_cap)
        reason_selected_pair_deleted = (
            reason_unary_candidate + reason_selected_pair_deleted
        ).clamp(-self.reason_cap, self.reason_cap)
        reason_control_pair_deleted = (
            reason_unary_candidate + reason_control_pair_deleted
        ).clamp(-self.reason_cap, self.reason_cap)
        reason_relevance = self.reason_traffic_mask.to(reason_candidate.dtype)[None]
        reason_selected_pair_deleted = reason_selected_pair_deleted * reason_relevance
        reason_control_pair_deleted = reason_control_pair_deleted * reason_relevance
        action_selected = action_attention.argmax(-1)
        reason_selected = reason_attention.argmax(-1)
        action_control = self._matched_control(action_attention, geometry["support"])
        reason_control = self._matched_control(reason_attention, geometry["support"])
        action_selected_deleted = self._deleted_candidate(
            self.action_encoder,
            self.action_output,
            self.action_cap,
            action_nodes,
            semantics,
            geometry["motion"],
            geometry["support"],
            geometry["valid_track"],
            geometry["interaction_risk"],
            role_probs,
            action_selected,
        )
        action_control_deleted = self._deleted_candidate(
            self.action_encoder,
            self.action_output,
            self.action_cap,
            action_nodes,
            semantics,
            geometry["motion"],
            geometry["support"],
            geometry["valid_track"],
            geometry["interaction_risk"],
            role_probs,
            action_control,
        )
        reason_selected_deleted = self._deleted_candidate(
            self.reason_encoder,
            self.reason_output,
            self.reason_cap,
            reason_nodes,
            semantics,
            geometry["motion"],
            geometry["support"],
            geometry["valid_track"],
            geometry["interaction_risk"],
            role_probs,
            reason_selected,
            self.reason_traffic_mask,
        )
        reason_control_deleted = self._deleted_candidate(
            self.reason_encoder,
            self.reason_output,
            self.reason_cap,
            reason_nodes,
            semantics,
            geometry["motion"],
            geometry["support"],
            geometry["valid_track"],
            geometry["interaction_risk"],
            role_probs,
            reason_control,
            self.reason_traffic_mask,
        )
        # Unary deletion must keep the pair route fixed; otherwise its measured
        # effect also removes pair evidence and overstates the unary route.
        action_selected_deleted = (
            action_selected_deleted + action_pair_candidate
        ).clamp(-self.action_cap, self.action_cap)
        action_control_deleted = (
            action_control_deleted + action_pair_candidate
        ).clamp(-self.action_cap, self.action_cap)
        reason_selected_deleted = (
            reason_selected_deleted + reason_pair_candidate
        ).clamp(-self.reason_cap, self.reason_cap) * reason_relevance
        reason_control_deleted = (
            reason_control_deleted + reason_pair_candidate
        ).clamp(-self.reason_cap, self.reason_cap) * reason_relevance
        # Candidates remain trainable while deploy stays at the exact baseline
        # until an out-of-forward train-calib proper-score audit opens a label.
        action_gate = self.action_deploy_gate.to(action_candidate)
        reason_gate = self.reason_deploy_gate.to(reason_candidate)
        action_delta = self._apply_deployment_policy(
            action_candidate, action_utility_gate, action_gate,
            self.action_deploy_scale.to(action_candidate),
            self.action_utility_cutoff.to(action_candidate), self.action_cap,
        )
        reason_delta = self._apply_deployment_policy(
            reason_candidate, reason_utility_gate, reason_gate,
            self.reason_deploy_scale.to(reason_candidate),
            self.reason_utility_cutoff.to(reason_candidate), self.reason_cap,
        )
        action_selected_mask = (
            (action_gate[None] > 0)
            & (action_utility_gate >= self.action_utility_cutoff.to(action_candidate)[None])
        ).to(action_candidate.dtype)
        reason_selected_mask = (
            (reason_gate[None] > 0)
            & (reason_utility_gate >= self.reason_utility_cutoff.to(reason_candidate)[None])
        ).to(reason_candidate.dtype)
        return {
            "object_intent_action_candidate": action_candidate,
            "object_intent_reason_candidate": reason_candidate,
            "object_intent_action_unary_candidate": action_unary_candidate,
            "object_intent_reason_unary_candidate": reason_unary_candidate,
            "object_intent_action_pair_candidate": action_pair_candidate,
            "object_intent_reason_pair_candidate": reason_pair_candidate,
            "object_intent_action_pair_attention": action_pair_attention,
            "object_intent_reason_pair_attention": reason_pair_attention,
            "object_intent_action_pair_support": action_pair_support,
            "object_intent_reason_pair_support": reason_pair_support,
            "object_intent_action_selected_pair": action_selected_pair,
            "object_intent_action_control_pair": action_control_pair,
            "object_intent_reason_selected_pair": reason_selected_pair,
            "object_intent_reason_control_pair": reason_control_pair,
            "object_intent_action_selected_pair_deleted_candidate": action_selected_pair_deleted,
            "object_intent_action_control_pair_deleted_candidate": action_control_pair_deleted,
            "object_intent_reason_selected_pair_deleted_candidate": reason_selected_pair_deleted,
            "object_intent_reason_control_pair_deleted_candidate": reason_control_pair_deleted,
            "object_intent_pair_min_future_distance": pair_geometry["min_future_distance"],
            "object_intent_pair_distance_reduction": pair_geometry["distance_reduction"],
            "object_intent_action_deploy_gate": action_gate[None].expand_as(action_candidate),
            "object_intent_reason_deploy_gate": reason_gate[None].expand_as(reason_candidate),
            "object_intent_action_deploy_scale": self.action_deploy_scale.to(action_candidate)[None].expand_as(action_candidate),
            "object_intent_reason_deploy_scale": self.reason_deploy_scale.to(reason_candidate)[None].expand_as(reason_candidate),
            "object_intent_action_utility_cutoff": self.action_utility_cutoff.to(action_candidate)[None].expand_as(action_candidate),
            "object_intent_reason_utility_cutoff": self.reason_utility_cutoff.to(reason_candidate)[None].expand_as(reason_candidate),
            "object_intent_action_utility_logit": action_utility_logit,
            "object_intent_reason_utility_logit": reason_utility_logit,
            "object_intent_action_utility_gate": action_utility_gate,
            "object_intent_reason_utility_gate": reason_utility_gate,
            "object_intent_action_utility_selected": action_selected_mask,
            "object_intent_reason_utility_selected": reason_selected_mask,
            "object_intent_action_delta": action_delta,
            "object_intent_reason_delta": reason_delta,
            "object_intent_action_attention": action_attention,
            "object_intent_reason_attention": reason_attention,
            "object_intent_action_semantic_attention": action_semantic_attention,
            "object_intent_action_motion_attention": action_motion_attention,
            "object_intent_reason_semantic_attention": reason_semantic_attention,
            "object_intent_reason_motion_attention": reason_motion_attention,
            "object_intent_action_motion_mix": action_motion_mix,
            "object_intent_reason_motion_mix": reason_motion_mix,
            "object_intent_action_support": action_support,
            "object_intent_reason_support": reason_support,
            "object_intent_action_selected_track": action_selected,
            "object_intent_action_control_track": action_control,
            "object_intent_reason_selected_track": reason_selected,
            "object_intent_reason_control_track": reason_control,
            "object_intent_action_selected_deleted_delta": self._apply_deployment_policy(action_selected_deleted, action_utility_gate, action_gate, self.action_deploy_scale.to(action_candidate), self.action_utility_cutoff.to(action_candidate), self.action_cap),
            "object_intent_action_control_deleted_delta": self._apply_deployment_policy(action_control_deleted, action_utility_gate, action_gate, self.action_deploy_scale.to(action_candidate), self.action_utility_cutoff.to(action_candidate), self.action_cap),
            "object_intent_reason_selected_deleted_delta": self._apply_deployment_policy(reason_selected_deleted, reason_utility_gate, reason_gate, self.reason_deploy_scale.to(reason_candidate), self.reason_utility_cutoff.to(reason_candidate), self.reason_cap),
            "object_intent_reason_control_deleted_delta": self._apply_deployment_policy(reason_control_deleted, reason_utility_gate, reason_gate, self.reason_deploy_scale.to(reason_candidate), self.reason_utility_cutoff.to(reason_candidate), self.reason_cap),
            "object_intent_action_selected_deleted_candidate": action_selected_deleted,
            "object_intent_action_control_deleted_candidate": action_control_deleted,
            "object_intent_reason_selected_deleted_candidate": reason_selected_deleted,
            "object_intent_reason_control_deleted_candidate": reason_control_deleted,
            "object_intent_track_semantics": semantics,
            "object_intent_track_semantics_by_frame": temporal_semantics,
            "object_intent_track_role_logits": role_logits,
            "object_intent_track_role_probs": role_probs,
            "object_intent_track_role_probs_by_frame": frame_role_probs,
            "object_intent_semantic_temporal_weights": semantic_temporal_weights,
            "object_intent_track_role_consistency": role_consistency,
            "object_intent_track_foreground_probability": foreground_probability,
            "object_intent_action_role_mass": torch.einsum(
                "blk,bkr->blr", action_attention, role_probs
            ),
            "object_intent_reason_role_mass": torch.einsum(
                "blk,bkr->blr", reason_attention, role_probs
            ),
            "object_intent_motion_features": geometry["motion"],
            "object_intent_future_xy": geometry["future_xy"],
            "object_intent_future_ego_distance": geometry["future_ego_distance"],
            "object_intent_future_approach_risk": geometry["future_approach_risk"],
            "object_intent_ego_relative_xy": geometry["ego_relative_xy"],
            "object_intent_track_support": geometry["support"],
            "object_intent_interaction_risk": geometry["interaction_risk"],
            "object_intent_common_displacement": geometry["common_displacement"],
            "object_intent_exclusive_displacement": geometry["exclusive_displacement"],
        }
