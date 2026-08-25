from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .acpr_sparse_ops import entmax15_bisect


def select_semantic_traffic_seeds(
    patch_tokens: torch.Tensor,
    predicate_attention: torch.Tensor,
    predicate_probability: torch.Tensor,
    *,
    grid_hw: tuple[int, int],
    topk: int,
) -> dict[str, torch.Tensor]:
    """Select unique patches supported by visible semantic predicates."""
    if patch_tokens.ndim != 3 or predicate_attention.ndim != 3:
        raise ValueError("patch tokens and predicate attention must be rank three")
    batch, patches, dim = patch_tokens.shape
    if predicate_attention.shape[0] != batch or predicate_attention.shape[2] != patches:
        raise ValueError("predicate attention does not match patch field")
    if predicate_probability.shape != predicate_attention.shape[:2]:
        raise ValueError("predicate_probability must be [B,P]")
    height, width = grid_hw
    if height * width != patches:
        raise ValueError("grid_hw does not match patch count")
    count = min(int(topk), patches)
    if count < 1:
        raise ValueError("topk must be positive")

    attention = predicate_attention.clamp_min(0.0)
    attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
    visible_support = attention * predicate_probability.detach().clamp(0.0, 1.0)[..., None]
    patch_score, predicate_id = visible_support.max(dim=1)
    score, index = patch_score.topk(count, dim=-1)
    weight = score / score.sum(-1, keepdim=True).clamp_min(1e-8)
    gathered = torch.gather(
        patch_tokens, 1, index[..., None].expand(-1, -1, dim)
    )
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=patch_tokens.device, dtype=patch_tokens.dtype),
        torch.linspace(-1.0, 1.0, width, device=patch_tokens.device, dtype=patch_tokens.dtype),
        indexing="ij",
    )
    coordinates = torch.stack((xx, yy), dim=-1).flatten(0, 1)
    return {
        "tokens": gathered[:, None],
        "xy": coordinates[index][:, None],
        "weights": weight[:, None],
        "indices": index,
        "predicate_ids": torch.gather(predicate_id, 1, index),
        "semantic_patch_score": score,
    }


class _TargetRelationalEncoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.appearance = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
        self.motion = nn.Sequential(nn.Linear(14, dim), nn.GELU(), nn.LayerNorm(dim))
        self.relation = nn.Sequential(nn.Linear(8, dim), nn.GELU(), nn.Linear(dim, dim))
        self.target_relation_query = nn.Linear(dim, dim, bias=False)
        self.target_relation_key = nn.Sequential(
            nn.Linear(8, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.target_relation_value = nn.Sequential(
            nn.Linear(8, dim), nn.GELU(), nn.LayerNorm(dim)
        )
        self.target_relation_output = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.target_relation_output.weight)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        target_nodes: torch.Tensor,
        appearance: torch.Tensor,
        motion_features: torch.Tensor,
        relation_features: torch.Tensor,
        relation_weight: torch.Tensor,
        support: torch.Tensor,
        track_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        relation_summary = torch.einsum(
            "bkj,bkjf->bkf", relation_weight, relation_features
        )
        base_track = (
            self.appearance(appearance)
            + self.motion(motion_features)
            + self.relation(relation_summary)
        )
        pair_score = torch.einsum(
            "bld,bkjd->blkj",
            self.target_relation_query(target_nodes),
            self.target_relation_key(relation_features),
        ) / math.sqrt(base_track.shape[-1])
        pair_valid = relation_weight > 0
        pair_score = pair_score + relation_weight.clamp_min(1e-8).log()[:, None]
        pair_score = pair_score.masked_fill(~pair_valid[:, None], -1e4)
        pair_attention = entmax15_bisect(pair_score, dim=-1)
        pair_attention = pair_attention * pair_valid[:, None].to(pair_attention.dtype)
        pair_attention = pair_attention / pair_attention.sum(-1, keepdim=True).clamp_min(1e-8)
        target_relation = torch.einsum(
            "blkj,bkjd->blkd",
            pair_attention,
            self.target_relation_value(relation_features),
        )
        target_relation = self.target_relation_output(target_relation)
        score = torch.einsum(
            "bld,blkd->blk",
            self.query(target_nodes),
            self.key(base_track)[:, None] + self.key(target_relation),
        ) / math.sqrt(base_track.shape[-1])
        score = score + support.clamp_min(1e-6).log()[:, None]
        if track_mask is not None:
            if track_mask.shape == support.shape:
                track_mask = track_mask[:, None].expand_as(score)
            elif track_mask.shape != score.shape:
                raise ValueError("track_mask must be [B,K] or [B,L,K]")
            score = score.masked_fill(~track_mask, -1e4)
        attention = entmax15_bisect(score, dim=-1)
        evidence = torch.einsum(
            "blk,blkd->bld",
            attention,
            self.value(base_track)[:, None] + self.value(target_relation),
        )
        transported_support = torch.einsum("blk,bk->bl", attention, support)
        return evidence, attention, transported_support, pair_attention


class TIDARelationalTrafficFlow(nn.Module):
    """Target-private relative traffic reasoning over ego-compensated tracks."""

    def __init__(
        self,
        dim: int = 384,
        num_actions: int = 4,
        num_reasons: int = 21,
        heads: int = 4,
        action_cap: float = 0.12,
        reason_cap: float = 0.10,
        reason_traffic_indices: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads or action_cap <= 0 or reason_cap <= 0:
            raise ValueError("invalid relational traffic dimensions")
        self.num_actions = int(num_actions)
        self.num_reasons = int(num_reasons)
        self.action_cap = float(action_cap)
        self.reason_cap = float(reason_cap)
        reason_mask = torch.ones(self.num_reasons, dtype=torch.bool)
        if reason_traffic_indices is not None:
            reason_mask.zero_()
            for index in reason_traffic_indices:
                if int(index) < 0 or int(index) >= self.num_reasons:
                    raise ValueError("reason traffic index is out of range")
                reason_mask[int(index)] = True
            if not reason_mask.any():
                raise ValueError("reason traffic mask cannot be empty")
        self.register_buffer("reason_traffic_mask", reason_mask)
        # No trainable tensor is shared across these branches: reason gradients
        # cannot alter action traffic evidence or its output geometry.
        self.action_encoder = _TargetRelationalEncoder(dim)
        self.reason_encoder = _TargetRelationalEncoder(dim)
        self.action_output = nn.Linear(dim, 1, bias=False)
        self.reason_output = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.action_output.weight)
        nn.init.zeros_(self.reason_output.weight)

    @staticmethod
    def _matched_control_track(
        target_attention: torch.Tensor, support: torch.Tensor,
    ) -> torch.Tensor:
        """Choose a low-credit track with visibility support matched to the selected one."""
        if target_attention.ndim != 3 or support.shape != (
            target_attention.shape[0], target_attention.shape[2]
        ):
            raise ValueError("target attention/support shape mismatch")
        selected = target_attention.argmax(-1)
        selected_support = torch.gather(
            support[:, None].expand(-1, target_attention.shape[1], -1),
            2,
            selected[..., None],
        ).squeeze(-1)
        support_distance = (support[:, None] - selected_support[..., None]).abs()
        attention_scale = target_attention.amax(-1, keepdim=True).clamp_min(1e-6)
        cost = support_distance + target_attention / attention_scale
        valid = support[:, None] > 0
        valid = valid.expand_as(cost).clone()
        valid.scatter_(2, selected[..., None], False)
        cost = cost.masked_fill(~valid, float("inf"))
        control = cost.argmin(-1)
        no_valid = ~valid.any(-1)
        fallback = (selected + 1) % target_attention.shape[-1]
        return torch.where(no_valid, fallback, control)

    @staticmethod
    def _geometry(
        trajectory_xy: torch.Tensor,
        trajectory_visibility: torch.Tensor,
        trajectory_pair_valid: torch.Tensor,
        exclusive_displacement: torch.Tensor,
        anchor_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        xy = trajectory_xy[:, 0]
        visibility = trajectory_visibility[:, 0]
        pair_valid = trajectory_pair_valid[:, 0].to(xy.dtype)
        velocity = exclusive_displacement[:, 0]
        denominator = pair_valid.sum(-1, keepdim=True).clamp_min(1.0)
        mean_velocity = (velocity * pair_valid[..., None]).sum(-2) / denominator
        last_velocity = velocity[..., -1, :]
        acceleration = torch.zeros_like(velocity)
        if velocity.shape[-2] > 1:
            acceleration[..., 1:, :] = velocity[..., 1:, :] - velocity[..., :-1, :]
        mean_acceleration = (acceleration * pair_valid[..., None]).sum(-2) / denominator
        speed = velocity.square().sum(-1).sqrt()
        mean_speed = (speed * pair_valid).sum(-1, keepdim=True) / denominator
        last_speed = speed[..., -1:]
        radial_velocity = (mean_velocity * xy[..., -1, :]).sum(-1, keepdim=True)
        visible_fraction = visibility.mean(-1, keepdim=True)
        path_length = (speed * pair_valid).sum(-1, keepdim=True)
        motion_features = torch.cat(
            (
                xy[..., -1, :], mean_velocity, last_velocity, mean_acceleration,
                mean_speed, last_speed, radial_velocity, visible_fraction,
                anchor_weight[:, 0, :, None], path_length,
            ),
            dim=-1,
        )

        final_xy = xy[..., -1, :]
        relative_position = final_xy[:, :, None] - final_xy[:, None, :]
        relative_velocity = mean_velocity[:, :, None] - mean_velocity[:, None, :]
        distance = relative_position.square().sum(-1).sqrt()
        closing = -(relative_position * relative_velocity).sum(-1) / distance.clamp_min(1e-4)
        crossing = (
            relative_position[..., 0] * relative_velocity[..., 1]
            - relative_position[..., 1] * relative_velocity[..., 0]
        ) / distance.clamp_min(1e-4)
        positive_closing = closing.relu()
        ttc_risk = positive_closing / (distance + positive_closing + 1e-4)
        relation_features = torch.cat(
            (
                relative_position, relative_velocity, distance[..., None],
                closing[..., None], crossing[..., None], ttc_risk[..., None],
            ),
            dim=-1,
        )
        tracks = distance.shape[-1]
        off_diagonal = ~torch.eye(tracks, dtype=torch.bool, device=distance.device)[None]
        valid_track = visibility.gt(0).any(-1)
        valid_relation = valid_track[:, :, None] & valid_track[:, None, :] & off_diagonal
        # Nearby pairs matter, but imminent closing/crossing pairs must not be
        # averaged away by equally distant static neighbours.
        dynamic_priority = 1.0 + 2.0 * ttc_risk + 0.5 * crossing.abs().clamp_max(1.0)
        relation_weight = (
            torch.exp(-2.0 * distance)
            * dynamic_priority
            * valid_relation.to(distance.dtype)
        )
        relation_weight = relation_weight / relation_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        # Seed weights are normalized over K and therefore are not confidence
        # probabilities. Using them directly attenuates every route as K grows.
        # Preserve their relative confidence while keeping support in [0, 1].
        anchor = anchor_weight[:, 0].clamp_min(0.0)
        relative_anchor = anchor / anchor.amax(-1, keepdim=True).clamp_min(1e-8)
        support = visibility.mean(-1) * relative_anchor.sqrt()
        return motion_features, relation_features, relation_weight, support

    def forward(
        self,
        action_nodes: torch.Tensor,
        reason_nodes: torch.Tensor,
        trajectory_appearance: torch.Tensor,
        trajectory_xy: torch.Tensor,
        trajectory_visibility: torch.Tensor,
        trajectory_pair_valid: torch.Tensor,
        common_displacement: torch.Tensor,
        exclusive_displacement: torch.Tensor,
        anchor_weight: torch.Tensor,
        *,
        track_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del common_displacement  # exclusive_displacement is already ego-motion compensated.
        if trajectory_appearance.ndim != 5 or trajectory_appearance.shape[1] != 1:
            raise ValueError("semantic trajectories must be [B,1,K,T,D]")
        appearance_weight = trajectory_visibility[:, 0]
        appearance_weight = appearance_weight / appearance_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        appearance = torch.einsum(
            "bkt,bktd->bkd", appearance_weight, trajectory_appearance[:, 0]
        )
        motion, relations, relation_weight, support = self._geometry(
            trajectory_xy, trajectory_visibility, trajectory_pair_valid,
            exclusive_displacement, anchor_weight,
        )
        action_evidence, action_attention, action_support, action_pair_attention = self.action_encoder(
            action_nodes, appearance, motion, relations, relation_weight, support, track_mask
        )
        reason_evidence, reason_attention, reason_support, reason_pair_attention = self.reason_encoder(
            reason_nodes, appearance, motion, relations, relation_weight, support, track_mask
        )
        def bounded_candidate(
            evidence: torch.Tensor, transported_support: torch.Tensor,
            output: nn.Linear, cap: float, relevance_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            candidate = float(cap) * transported_support * torch.tanh(
                output(evidence).squeeze(-1)
            )
            # Traffic evidence may reorder targets, but cannot emulate a
            # global threshold shift shared by every label.
            if relevance_mask is None:
                return candidate - candidate.mean(-1, keepdim=True)
            mask = relevance_mask.to(candidate.dtype)[None]
            active_mean = (candidate * mask).sum(-1, keepdim=True) / mask.sum().clamp_min(1.0)
            return (candidate - active_mean) * mask

        action_candidate = bounded_candidate(
            action_evidence, action_support, self.action_output, self.action_cap
        )
        reason_candidate = bounded_candidate(
            reason_evidence, reason_support, self.reason_output, self.reason_cap,
            self.reason_traffic_mask,
        )
        tracks = support.shape[1]
        if tracks < 2:
            raise ValueError("relational deletion contrast requires at least two tracks")
        action_selected_track = action_attention.argmax(-1)
        reason_selected_track = reason_attention.argmax(-1)
        action_random_track = self._matched_control_track(action_attention, support)
        reason_random_track = self._matched_control_track(reason_attention, support)

        def deletion_mask(index: torch.Tensor) -> torch.Tensor:
            mask = torch.ones_like(
                index[..., None].expand(-1, -1, tracks), dtype=torch.bool
            )
            return mask.scatter(2, index[..., None], False)

        action_selected_mask = deletion_mask(action_selected_track)
        reason_selected_mask = deletion_mask(reason_selected_track)
        action_random_mask = deletion_mask(action_random_track)
        reason_random_mask = deletion_mask(reason_random_track)

        def deleted(
            action_mask: torch.Tensor, reason_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            deleted_action, _, deleted_action_support, _ = self.action_encoder(
                action_nodes, appearance, motion, relations, relation_weight, support,
                action_mask,
            )
            deleted_reason, _, deleted_reason_support, _ = self.reason_encoder(
                reason_nodes, appearance, motion, relations, relation_weight, support,
                reason_mask,
            )
            return (
                bounded_candidate(
                    deleted_action, deleted_action_support, self.action_output, self.action_cap
                ),
                bounded_candidate(
                    deleted_reason, deleted_reason_support, self.reason_output, self.reason_cap,
                    self.reason_traffic_mask,
                ),
            )

        selected_action_delta, selected_reason_delta = deleted(
            action_selected_mask, reason_selected_mask
        )
        random_action_delta, random_reason_delta = deleted(
            action_random_mask, reason_random_mask
        )
        # Case-level summaries are retained for compact visualization. The
        # complete per-target routes below are the source of audit metrics.
        target_mass = torch.cat((action_attention, reason_attention), dim=1).mean(1)
        selected_track = target_mass.argmax(-1)
        random_track = self._matched_control_track(target_mass[:, None], support)[:, 0]
        interaction_risk = relations[..., -1]
        return {
            "relational_action_delta": action_candidate,
            "relational_reason_delta": reason_candidate,
            "relational_action_candidate": action_candidate,
            "relational_reason_candidate": reason_candidate,
            "relational_action_selected_deleted_delta": selected_action_delta,
            "relational_action_random_deleted_delta": random_action_delta,
            "relational_reason_selected_deleted_delta": selected_reason_delta,
            "relational_reason_random_deleted_delta": random_reason_delta,
            "relational_selected_track": selected_track,
            "relational_random_track": random_track,
            "relational_action_selected_track": action_selected_track,
            "relational_action_random_track": action_random_track,
            "relational_reason_selected_track": reason_selected_track,
            "relational_reason_random_track": reason_random_track,
            "relational_action_attention": action_attention,
            "relational_reason_attention": reason_attention,
            "relational_action_pair_attention": action_pair_attention,
            "relational_reason_pair_attention": reason_pair_attention,
            "relational_action_support": action_support,
            "relational_reason_support": reason_support,
            "relational_track_support": support,
            "relational_motion_features": motion,
            "relational_pair_features": relations,
            "relational_pair_weights": relation_weight,
            "relational_interaction_risk": interaction_risk,
        }
