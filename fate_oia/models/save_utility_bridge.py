from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .acpr_sparse_ops import entmax15_bisect


SAVE_UTILITY_RANK = 32
SAVE_UTILITY_TEACHER_INTERVAL_UPDATES = 4
SAVE_UTILITY_MAX_TEACHER_SAMPLES = 2
SAVE_UTILITY_MAX_TEACHER_PREDICATES = 2
SAVE_UTILITY_MAX_SELECTED_PATCHES = 24
SAVE_UTILITY_MIN_SELECTED_PATCHES = 8


def _require_tensor(value: Tensor, name: str, ndim: int) -> Tensor:
    if not isinstance(value, Tensor) or value.ndim != ndim:
        raise ValueError(f"{name} must be a rank-{ndim} tensor")
    return value


def _real_candidate_weight(candidate_weight: Tensor, factor_dim: int) -> Tensor:
    candidate_weight = _require_tensor(candidate_weight, "candidate_weight", 3)
    if candidate_weight.shape[-1] == factor_dim + 1:
        return candidate_weight[..., :factor_dim]
    if candidate_weight.shape[-1] == factor_dim:
        return candidate_weight
    raise ValueError(
        f"candidate_weight must end in {factor_dim} or {factor_dim + 1}, "
        f"got {candidate_weight.shape[-1]}"
    )


def select_teacher_predicates(
    candidate_weight: Tensor,
    predicate_reliability: Tensor,
    base_predicate_overlap: Tensor,
    *,
    max_predicates: int = SAVE_UTILITY_MAX_TEACHER_PREDICATES,
    utility_logit: Tensor | None = None,
) -> Tensor:
    """Select teacher predicates without consulting the utility predictor.

    The optional ``utility_logit`` argument is accepted only so callers can
    pass diagnostic predictions without creating a second selection path.  It
    is deliberately ignored: selected candidates are always ranked by
    candidate weight * reliability * base overlap.
    """
    del utility_logit
    candidate_weight = _require_tensor(candidate_weight, "candidate_weight", 3)
    predicate_reliability = _require_tensor(
        predicate_reliability, "predicate_reliability", 2
    )
    base_predicate_overlap = _require_tensor(
        base_predicate_overlap, "base_predicate_overlap", 3
    )
    batch, actions, factors = base_predicate_overlap.shape
    if candidate_weight.shape[:2] != (batch, actions):
        raise ValueError("candidate_weight and base_predicate_overlap batch/action mismatch")
    real = _real_candidate_weight(candidate_weight, factors).float().clamp_min(0.0)
    reliability = predicate_reliability.float()
    if tuple(reliability.shape) != (batch, factors):
        raise ValueError("predicate_reliability must have shape [B,F]")
    overlap = base_predicate_overlap.float().clamp_min(0.0)
    score = real * reliability.unsqueeze(1) * overlap
    count = min(max(1, int(max_predicates)), factors)
    values, indices = torch.topk(score, k=count, dim=-1, largest=True, sorted=True)
    return indices.masked_fill(values <= 0.0, -1)


def _stable_action_choice(
    logits: Tensor,
    targets: Tensor,
) -> tuple[Tensor, Tensor]:
    signed_margin = (targets.float() * 2.0 - 1.0) * logits.float()
    selected: list[int] = []
    priority: list[float] = []
    for row_logits, row_targets, row_margin in zip(logits, targets, signed_margin):
        positive = torch.nonzero(row_targets > 0.5, as_tuple=False).flatten()
        if positive.numel() > 0:
            order = sorted(
                (int(index) for index in positive),
                key=lambda index: (float(row_margin[index]), index),
            )
            action = order[0]
            score = float(row_margin[action])
        else:
            order = sorted(
                range(row_logits.numel()),
                key=lambda index: (abs(float(row_logits[index])), index),
            )
            action = order[0]
            score = abs(float(row_logits[action]))
        selected.append(action)
        priority.append(score)
    if not selected:
        empty = logits.new_empty((0,), dtype=torch.long)
        return empty, logits.new_empty((0,))
    return (
        torch.tensor(selected, dtype=torch.long, device=logits.device),
        logits.new_tensor(priority),
    )


def select_sparse_teacher_targets(
    base_action_logits: Tensor,
    action_targets: Tensor,
    *,
    max_samples: int = SAVE_UTILITY_MAX_TEACHER_SAMPLES,
) -> dict[str, Tensor]:
    """Choose hard samples and one action per sample from the base branch."""
    base_action_logits = _require_tensor(base_action_logits, "base_action_logits", 2)
    action_targets = _require_tensor(action_targets, "action_targets", 2)
    if base_action_logits.shape != action_targets.shape:
        raise ValueError("base_action_logits and action_targets must have the same shape")
    action_indices, priority = _stable_action_choice(
        base_action_logits,
        action_targets,
    )
    sample_indices = torch.arange(
        base_action_logits.shape[0], device=base_action_logits.device, dtype=torch.long
    )
    order = sorted(
        range(sample_indices.numel()),
        key=lambda index: (float(priority[index]), int(sample_indices[index])),
    )
    keep = order[: max(0, int(max_samples))]
    if not keep:
        empty = torch.empty(0, dtype=torch.long, device=base_action_logits.device)
        return {
            "sample_indices": empty,
            "action_indices": empty,
            "target_margin": base_action_logits.new_empty((0,)),
        }
    selected_samples = sample_indices[keep]
    selected_actions = action_indices[keep]
    signed = (action_targets * 2.0 - 1.0) * base_action_logits
    selected_margin = signed[selected_samples, selected_actions]
    return {
        "sample_indices": selected_samples,
        "action_indices": selected_actions,
        "target_margin": selected_margin,
    }


def build_sparse_teacher_plan(
    base_action_logits: Tensor,
    action_targets: Tensor,
    candidate_weight: Tensor,
    predicate_reliability: Tensor,
    base_predicate_overlap: Tensor,
    *,
    utility_logit: Tensor | None = None,
    max_samples: int = SAVE_UTILITY_MAX_TEACHER_SAMPLES,
    max_predicates: int = SAVE_UTILITY_MAX_TEACHER_PREDICATES,
) -> dict[str, Any]:
    """Build the bounded sparse teacher queue without self-selection."""
    del utility_logit
    targets = select_sparse_teacher_targets(
        base_action_logits,
        action_targets,
        max_samples=max_samples,
    )
    selected = select_teacher_predicates(
        candidate_weight,
        predicate_reliability,
        base_predicate_overlap,
        max_predicates=max_predicates,
    )
    selected_samples = targets["sample_indices"]
    selected_actions = targets["action_indices"]
    if selected_samples.numel() == 0:
        predicate_indices = torch.empty(
            (0, min(max(1, int(max_predicates)), base_predicate_overlap.shape[-1])),
            dtype=torch.long,
            device=base_action_logits.device,
        )
    else:
        predicate_indices = selected[selected_samples, selected_actions]

    valid = predicate_indices >= 0
    sample_matrix = selected_samples.unsqueeze(-1).expand_as(predicate_indices)
    action_matrix = selected_actions.unsqueeze(-1).expand_as(predicate_indices)
    sample_indices = sample_matrix[valid]
    action_indices = action_matrix[valid]
    factor_indices = predicate_indices[valid]
    factors = base_predicate_overlap.shape[-1]
    real = _real_candidate_weight(candidate_weight, factors).float()
    if factor_indices.numel() == 0:
        selected_candidate_weight = base_action_logits.new_empty((0,))
        selected_reliability = base_action_logits.new_empty((0,))
        selected_overlap = base_action_logits.new_empty((0,))
    else:
        selected_candidate_weight = real[
            sample_indices, action_indices, factor_indices
        ]
        selected_reliability = predicate_reliability.float()[
            sample_indices, factor_indices
        ]
        selected_overlap = base_predicate_overlap.float().clamp_min(0.0)[
            sample_indices, action_indices, factor_indices
        ]
    candidate_scores = (
        selected_candidate_weight * selected_reliability * selected_overlap
    )
    return {
        "selected_sample_indices": selected_samples,
        "selected_action_indices": selected_actions,
        "selected_target_margin": targets["target_margin"],
        "predicate_indices": predicate_indices,
        "sample_indices": sample_indices,
        "action_indices": action_indices,
        "factor_indices": factor_indices,
        "candidate_scores": candidate_scores,
        "selected_candidate_weight": selected_candidate_weight,
        "selected_reliability": selected_reliability,
        "selected_base_overlap": selected_overlap,
        "teacher_sample_indices": sample_indices,
        "teacher_action_indices": action_indices,
        "teacher_factor_indices": factor_indices,
        "selection_source": "candidate_weight*reliability*base_overlap",
        "utility_predictor_used_for_selection": False,
    }


def _grid_coordinates(
    count: int,
    grid_hw: tuple[int, int],
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    height, width = (int(grid_hw[0]), int(grid_hw[1]))
    if height <= 0 or width <= 0 or count != height * width:
        raise ValueError(f"grid_hw {grid_hw} does not match patch count {count}")
    indices = torch.arange(count, device=device)
    rows = torch.div(indices, width, rounding_mode="floor")
    columns = indices.remainder(width)
    sides = (columns * 3 // width).clamp_max(2)
    depths = (rows * 5 // height).clamp_max(4)
    sector = sides * 5 + depths
    return rows, columns, sector


def select_selected_patches(
    predicate_map: Tensor,
    predicate_indices: Tensor,
    *,
    candidate_scores: Tensor | None = None,
    max_patches: int = SAVE_UTILITY_MAX_SELECTED_PATCHES,
    mass_fraction: float = 0.60,
    min_patches: int = SAVE_UTILITY_MIN_SELECTED_PATCHES,
) -> Tensor:
    """Select a small deterministic patch set covering the chosen map mass."""
    predicate_map = _require_tensor(predicate_map, "predicate_map", 2).float().clamp_min(0.0)
    predicate_indices = torch.as_tensor(
        predicate_indices, device=predicate_map.device, dtype=torch.long
    ).reshape(-1)
    if predicate_indices.numel() == 0:
        raise ValueError("at least one predicate is required for selected deletion")
    valid = predicate_indices[(predicate_indices >= 0) & (predicate_indices < predicate_map.shape[0])]
    if valid.numel() == 0:
        raise ValueError("predicate_indices contains no valid predicate")
    if candidate_scores is None:
        weights = predicate_map.new_ones(valid.shape)
    else:
        scores = torch.as_tensor(candidate_scores, device=predicate_map.device).float().reshape(-1)
        if scores.numel() != predicate_indices.numel():
            raise ValueError("candidate_scores must match predicate_indices")
        weights = scores[(predicate_indices >= 0) & (predicate_indices < predicate_map.shape[0])]
        weights = weights.clamp_min(0.0)
        if float(weights.sum()) == 0.0:
            weights = predicate_map.new_ones(valid.shape)
    evidence = (predicate_map[valid] * weights.unsqueeze(-1)).sum(dim=0)
    if float(evidence.sum()) <= 0.0:
        evidence = predicate_map.new_ones((predicate_map.shape[-1],))
    order = torch.argsort(evidence, descending=True, stable=True)
    mass = evidence[order].cumsum(0) / evidence.sum().clamp_min(1e-8)
    covered = torch.nonzero(mass >= float(mass_fraction), as_tuple=False)
    required = int(covered[0, 0].item() + 1) if covered.numel() else order.numel()
    upper = min(int(max_patches), order.numel())
    lower = min(int(min_patches), upper)
    count = min(upper, max(lower, required))
    return order[:count]


def _action_patch_vector(action_contribution: Tensor | None, patches: int) -> Tensor | None:
    if action_contribution is None:
        return None
    value = torch.as_tensor(action_contribution).float()
    if value.ndim == 1 and value.shape[0] == patches:
        return value
    if value.ndim == 2 and value.shape[-1] == patches:
        return value.abs().mean(dim=0)
    if value.ndim == 3 and value.shape[-1] == patches:
        return value.abs().mean(dim=(0, 1))
    raise ValueError("action_contribution must end in the patch dimension")


def select_matched_control_patches(
    detail_field: Tensor,
    selected: Tensor,
    *,
    grid_hw: tuple[int, int] = (45, 80),
    predicate_map: Tensor | None = None,
    action_contribution: Tensor | None = None,
    valid_mask: Tensor | None = None,
    feature_norm_tolerance: float = 0.20,
    texture_tolerance: float = 0.25,
    predicate_overlap_tolerance: float = 0.25,
    y_tolerance: float | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Select a deterministic, disjoint control satisfying the SAVE contract."""
    if isinstance(detail_field, Tensor) and detail_field.ndim == 1:
        detail_field = detail_field.unsqueeze(-1)
    detail_field = _require_tensor(detail_field, "detail_field", 2).float()
    patches = detail_field.shape[0]
    selected = torch.as_tensor(selected, device=detail_field.device, dtype=torch.long).reshape(-1)
    if selected.numel() == 0 or selected.unique().numel() != selected.numel():
        raise ValueError("selected patches must be non-empty and unique")
    if bool((selected < 0).any()) or bool((selected >= patches).any()):
        raise ValueError("selected patches are out of range")
    _, _, sectors = _grid_coordinates(patches, grid_hw, device=detail_field.device)
    selected_sector = sectors[selected]
    selected_rows = torch.div(selected, int(grid_hw[1]), rounding_mode="floor").float()
    selected_y = selected_rows.mean()
    if y_tolerance is None:
        y_tolerance = max(2.0, float(grid_hw[0]) * 0.15)

    valid = torch.ones(patches, dtype=torch.bool, device=detail_field.device)
    if valid_mask is not None:
        valid = torch.as_tensor(valid_mask, device=detail_field.device).bool().reshape(-1)
        if valid.shape[0] != patches:
            raise ValueError("valid_mask must match the patch count")
    selected_mask = torch.zeros(patches, dtype=torch.bool, device=detail_field.device)
    selected_mask[selected] = True
    rows = torch.div(
        torch.arange(patches, device=detail_field.device),
        int(grid_hw[1]),
        rounding_mode="floor",
    ).float()
    same_sector = torch.isin(sectors, selected_sector)
    candidate = (
        valid
        & ~selected_mask
        & same_sector
        & ((rows - selected_y).abs() <= float(y_tolerance))
    )

    feature_norm = detail_field.norm(dim=-1)
    texture = detail_field.var(dim=-1, unbiased=False)
    selected_norm = feature_norm[selected].mean()
    selected_texture = texture[selected].mean()
    norm_delta = (feature_norm - selected_norm).abs() / selected_norm.abs().clamp_min(1e-6)
    texture_delta = (texture - selected_texture).abs() / selected_texture.abs().clamp_min(1e-6)
    candidate &= norm_delta <= float(feature_norm_tolerance)
    candidate &= texture_delta <= float(texture_tolerance)

    predicate_overlap = torch.zeros(patches, device=detail_field.device)
    if predicate_map is not None:
        maps = _require_tensor(predicate_map, "predicate_map", 2).float().clamp_min(0.0)
        if maps.shape[-1] != patches:
            raise ValueError("predicate_map must match the patch count")
        selected_profile = maps[:, selected].mean(dim=-1)
        candidate_profiles = maps.transpose(0, 1)
        predicate_overlap = F.cosine_similarity(
            candidate_profiles,
            selected_profile.unsqueeze(0).expand(patches, -1),
            dim=-1,
            eps=1e-8,
        ).clamp_min(0.0)
        candidate &= predicate_overlap <= float(predicate_overlap_tolerance)

    action_vector = _action_patch_vector(action_contribution, patches)
    if action_vector is None:
        action_vector = detail_field.new_zeros((patches,))
    selected_action = action_vector[selected].abs().mean()
    candidate &= action_vector.abs() <= selected_action + 1e-6

    score = (
        norm_delta
        + texture_delta
        + predicate_overlap
        + action_vector.abs() / selected_action.clamp_min(1e-6)
        + (rows - selected_y).abs() / max(float(grid_hw[0]), 1.0)
    )
    score = score + torch.arange(patches, device=detail_field.device).float() * 1e-8
    controls: list[Tensor] = []
    selected_sectors, sector_counts = torch.unique(
        selected_sector,
        sorted=True,
        return_counts=True,
    )
    for sector_value, required_count in zip(selected_sectors, sector_counts):
        sector_candidate = candidate & (sectors == sector_value)
        required = int(required_count.item())
        if int(sector_candidate.sum()) < required:
            raise ValueError(
                "insufficient exact matched controls: equal count, sector, y, "
                "feature norm, texture, predicate overlap, and low action effect are required"
            )
        controls.append(
            torch.topk(
                score.masked_fill(~sector_candidate, float("inf")),
                k=required,
                largest=False,
                sorted=True,
            ).indices
        )
    control = torch.cat(controls)
    count = selected.numel()
    control_sector = sectors[control]
    control_norm = feature_norm[control].mean()
    control_texture = texture[control].mean()
    control_action = action_vector[control].abs().mean()
    selected_side = int((selected_sector // 5).mode().values.item())
    control_side = int((control_sector // 5).mode().values.item())
    selected_depth = int((selected_sector.remainder(5)).mode().values.item())
    control_depth = int((control_sector.remainder(5)).mode().values.item())
    metadata: dict[str, Any] = {
        "selected_count": int(count),
        "control_count": int(control.numel()),
        "selected_sector": selected_sector.detach().cpu().tolist(),
        "control_sector": control_sector.detach().cpu().tolist(),
        "selected_side": selected_side,
        "control_side": control_side,
        "selected_depth_bin": selected_depth,
        "control_depth_bin": control_depth,
        "same_sector": bool(
            torch.equal(
                torch.bincount(control_sector, minlength=15),
                torch.bincount(selected_sector, minlength=15),
            )
        ),
        "similar_y": bool((rows[control] - selected_y).abs().max() <= float(y_tolerance)),
        "control_valid_fraction": float(valid[control].float().mean()),
        "feature_norm_relative_difference": float(
            (control_norm - selected_norm).abs() / selected_norm.abs().clamp_min(1e-6)
        ),
        "texture_variance_relative_difference": float(
            (control_texture - selected_texture).abs()
            / selected_texture.abs().clamp_min(1e-6)
        ),
        "predicate_overlap": float(predicate_overlap[control].mean()),
        "selected_action_contribution": float(selected_action),
        "control_action_contribution": float(control_action),
        "overlap_count": int(torch.isin(control, selected).sum()),
        "selection_method": "deterministic_matched_control",
    }
    if metadata["overlap_count"] != 0 or not metadata["same_sector"]:
        raise RuntimeError("matched control contract was violated")
    return control, metadata


select_geometry_matched_control = select_matched_control_patches


def _summary_to_dim(summary: Tensor, dim: int) -> Tensor:
    if summary.shape[-1] == dim:
        return summary
    # State summaries are allowed to expose a different number of ontology
    # states; reduce them to a scalar and broadcast rather than creating a
    # per-factor per-patch feature tensor.
    return summary.float().mean(dim=-1, keepdim=True).expand(*summary.shape[:-1], dim)


class SAVEUtilityBridge(nn.Module):
    """Action-validated utility predictor and sparse counterfactual teacher."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        factor_dim: int = 21,
        rank: int = SAVE_UTILITY_RANK,
        hidden_dim: int = 64,
        utility_teacher_interval_updates: int = SAVE_UTILITY_TEACHER_INTERVAL_UPDATES,
    ) -> None:
        super().__init__()
        if int(rank) != SAVE_UTILITY_RANK:
            raise ValueError("SAVE utility bridge requires rank-32 bilinear interactions")
        if int(utility_teacher_interval_updates) <= 0:
            raise ValueError("utility_teacher_interval_updates must be positive")
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.factor_dim = int(factor_dim)
        self.rank = int(rank)
        self.utility_rank = int(rank)
        self.utility_teacher_interval_updates = int(utility_teacher_interval_updates)

        self.candidate_query = nn.Linear(self.dim, self.dim, bias=False)
        self.candidate_key = nn.Linear(self.dim, self.dim, bias=False)
        self.null_candidate_key = nn.Parameter(torch.zeros(self.dim))
        self.candidate_bias = nn.Parameter(torch.zeros(self.action_dim, self.factor_dim + 1))

        self.state_projection = nn.Linear(self.dim, self.dim)
        self.utility_action_projection = nn.Linear(self.dim, self.rank, bias=False)
        self.utility_predicate_projection = nn.Linear(self.dim, self.rank, bias=False)
        self.utility_mlp = nn.Sequential(
            nn.Linear(self.rank + 4, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.null_utility_head = nn.Linear(self.dim, 1)
        self.register_buffer(
            "optimizer_updates",
            torch.zeros((), dtype=torch.long),
            persistent=True,
        )

    @property
    def utility_predictor(self) -> nn.Module:
        return self.utility_mlp

    @property
    def teacher_interval_updates(self) -> int:
        return self.utility_teacher_interval_updates

    @staticmethod
    def select_teacher_predicates(*args: Any, **kwargs: Any) -> Tensor:
        return select_teacher_predicates(*args, **kwargs)

    @staticmethod
    def select_sparse_teacher_targets(*args: Any, **kwargs: Any) -> dict[str, Tensor]:
        return select_sparse_teacher_targets(*args, **kwargs)

    @staticmethod
    def select_matched_control_patches(*args: Any, **kwargs: Any) -> tuple[Tensor, dict[str, Any]]:
        return select_matched_control_patches(*args, **kwargs)

    @staticmethod
    def build_sparse_teacher_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return build_sparse_teacher_plan(*args, **kwargs)

    def _candidate_weights(
        self,
        action_global_token: Tensor,
        predicate_token: Tensor,
    ) -> Tensor:
        query = self.candidate_query(action_global_token)
        key = self.candidate_key(predicate_token)
        real_score = torch.einsum("bad,bfd->baf", query, key) / math.sqrt(self.dim)
        null_score = torch.einsum("bad,d->ba", query, self.null_candidate_key)
        scores = torch.cat(
            (real_score, null_score.unsqueeze(-1)),
            dim=-1,
        ) + self.candidate_bias.unsqueeze(0)
        # Entmax is applied to the small candidate axis only.  The full patch
        # field never enters this route.
        return entmax15_bisect(scores.float(), dim=-1).to(action_global_token.dtype)

    def _utility_logits(
        self,
        action_global_token: Tensor,
        predicate_token: Tensor,
        predicate_state_summary: Tensor,
        predicate_reliability: Tensor,
        base_predicate_overlap: Tensor,
        global_detail_query_similarity: Tensor,
        candidate_weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        state = _summary_to_dim(predicate_state_summary, self.dim)
        state = state.to(self.state_projection.weight.dtype)
        predicate = predicate_token + self.state_projection(state)
        action_rank = self.utility_action_projection(action_global_token)
        predicate_rank = self.utility_predicate_projection(predicate)
        bilinear = action_rank.unsqueeze(2) * predicate_rank.unsqueeze(1)

        similarity = global_detail_query_similarity
        if similarity.ndim == 2:
            similarity = similarity.unsqueeze(-1).expand(
                -1, -1, self.factor_dim
            )
        if tuple(similarity.shape) != tuple(base_predicate_overlap.shape):
            raise ValueError("global_detail_query_similarity must have shape [B,A] or [B,A,F]")
        real_candidate = candidate_weight[..., : self.factor_dim]
        scalars = torch.stack(
            (
                predicate_reliability.unsqueeze(1).expand(-1, self.action_dim, -1),
                base_predicate_overlap,
                similarity,
                real_candidate,
            ),
            dim=-1,
        )
        utility_input = torch.cat((bilinear, scalars), dim=-1)
        mlp_dtype = self.utility_mlp[0].weight.dtype
        utility_logit = self.utility_mlp(utility_input.to(mlp_dtype)).squeeze(-1)
        null_logit = self.null_utility_head(action_global_token).squeeze(-1)
        with_null = torch.cat((utility_logit, null_logit.unsqueeze(-1)), dim=-1)
        return utility_logit, torch.sigmoid(utility_logit), with_null, torch.sigmoid(with_null)

    def forward(
        self,
        action_global_token: Tensor,
        predicate_token: Tensor,
        predicate_state_summary: Tensor | None = None,
        predicate_state_prob: Tensor | None = None,
        predicate_reliability: Tensor | None = None,
        predicate_visual_reliability: Tensor | None = None,
        base_predicate_overlap: Tensor | None = None,
        global_detail_query_similarity: Tensor | None = None,
        query_similarity: Tensor | None = None,
        *,
        base_action_predicate_overlap: Tensor | None = None,
        base_action_predicate_map_overlap: Tensor | None = None,
        global_detail_similarity: Tensor | None = None,
        detail_field: Tensor | None = None,
        predicate_map: Tensor | None = None,
        action_contribution: Tensor | None = None,
        base_action_logits: Tensor | None = None,
        action_targets: Tensor | None = None,
        action_named_contribution: Tensor | None = None,
        optimizer_update: int | None = None,
        run_teacher: bool | None = None,
        utility_logit_for_teacher: Tensor | None = None,
        teacher_decoder: Callable[..., Any] | None = None,
        teacher_grid_hw: tuple[int, int] = (45, 80),
    ) -> dict[str, Any]:
        if base_predicate_overlap is None:
            base_predicate_overlap = base_action_predicate_overlap
        if base_predicate_overlap is None:
            base_predicate_overlap = base_action_predicate_map_overlap
        if global_detail_query_similarity is None:
            global_detail_query_similarity = global_detail_similarity
        if global_detail_query_similarity is None:
            global_detail_query_similarity = query_similarity
        action_global_token = _require_tensor(
            action_global_token, "action_global_token", 3
        )
        predicate_token = _require_tensor(predicate_token, "predicate_token", 3)
        if tuple(action_global_token.shape[1:]) != (self.action_dim, self.dim):
            raise ValueError("action_global_token must have shape [B,A,D]")
        if tuple(predicate_token.shape[1:]) != (self.factor_dim, self.dim):
            raise ValueError("predicate_token must have shape [B,F,D]")
        model_dtype = self.candidate_query.weight.dtype
        action_global_token = action_global_token.to(model_dtype)
        predicate_token = predicate_token.to(model_dtype)
        batch = action_global_token.shape[0]
        if predicate_state_summary is None:
            predicate_state_summary = predicate_state_prob
        if predicate_state_summary is None:
            predicate_state_summary = predicate_token
        else:
            predicate_state_summary = predicate_state_summary.to(model_dtype)
        if predicate_reliability is None:
            predicate_reliability = predicate_visual_reliability
        if predicate_reliability is None:
            predicate_reliability = action_global_token.new_ones((batch, self.factor_dim))
        if base_predicate_overlap is None:
            base_predicate_overlap = action_global_token.new_zeros(
                (batch, self.action_dim, self.factor_dim)
            )
        if global_detail_query_similarity is None:
            global_detail_query_similarity = action_global_token.new_zeros(
                (batch, self.action_dim, self.factor_dim)
            )
        if tuple(predicate_reliability.shape) != (batch, self.factor_dim):
            raise ValueError("predicate_reliability must have shape [B,F]")
        if tuple(base_predicate_overlap.shape) != (batch, self.action_dim, self.factor_dim):
            raise ValueError("base_predicate_overlap must have shape [B,A,F]")

        candidate_weight = self._candidate_weights(
            action_global_token,
            predicate_token,
        )
        utility_logit, utility_prob, utility_with_null, utility_prob_with_null = self._utility_logits(
            action_global_token,
            predicate_token,
            predicate_state_summary,
            predicate_reliability.to(model_dtype),
            base_predicate_overlap.to(model_dtype),
            global_detail_query_similarity.to(model_dtype),
            candidate_weight.float(),
        )

        teacher_plan: dict[str, Any] | None = None
        teacher_due = False
        if optimizer_update is not None:
            update = int(optimizer_update)
            teacher_due = (
                self.training
                and update > 0
                and update % self.utility_teacher_interval_updates == 0
            )
            if teacher_due:
                with torch.no_grad():
                    self.optimizer_updates.fill_(update)
        if teacher_due:
            if run_teacher is False:
                raise RuntimeError("due counterfactual teacher cannot be disabled during training")
            required = {
                "detail_field": detail_field,
                "predicate_map": predicate_map,
                "action_contribution": action_contribution,
                "base_action_logits": base_action_logits,
                "action_targets": action_targets,
                "teacher_decoder": teacher_decoder,
            }
            missing = [name for name, value in required.items() if value is None]
            if missing:
                raise RuntimeError(
                    "due counterfactual teacher is missing required inputs: "
                    + ", ".join(missing)
                )
            teacher_plan = build_sparse_counterfactual_teacher(
                base_action_logits,
                action_targets,
                candidate_weight,
                predicate_reliability,
                base_predicate_overlap,
                detail_field=detail_field,
                predicate_map=predicate_map,
                action_contribution=action_contribution,
                grid_hw=teacher_grid_hw,
                utility_logit=utility_logit,
                teacher_decoder=teacher_decoder,
            )

        return {
            "predicate_candidate_weight": candidate_weight,
            "predicate_candidate_weight_real": candidate_weight[..., : self.factor_dim],
            "predicate_candidate_weight_null": candidate_weight[..., self.factor_dim],
            "utility_logit": utility_logit,
            "utility_prob": utility_prob,
            "utility_logit_with_null": utility_with_null,
            "utility_prob_with_null": utility_prob_with_null,
            "utility_rank": torch.tensor(self.rank, device=utility_logit.device),
            "utility_teacher_due": teacher_due,
            "teacher_plan": teacher_plan,
            "utility_teacher_target": (
                None if teacher_plan is None else teacher_plan.get("utility_teacher_target")
            ),
            "utility_teacher_prediction": (
                None if teacher_plan is None else teacher_plan.get("utility_teacher_prediction")
            ),
            "utility_teacher_sample_indices": (
                None if teacher_plan is None else teacher_plan.get("sample_indices")
            ),
            "utility_teacher_action_indices": (
                None if teacher_plan is None else teacher_plan.get("action_indices")
            ),
            "utility_teacher_factor_indices": (
                None if teacher_plan is None else teacher_plan.get("factor_indices")
            ),
            "utility_counterfactual_weight": utility_logit.new_tensor(0.10),
            "utility_dense_auxiliary_weight": utility_logit.new_tensor(0.02),
            "action_named_contribution": action_named_contribution,
        }


def _teacher_action_logits(
    decoded: Any,
    *,
    variant: str,
    action_dim: int,
    reference: Tensor,
) -> Tensor:
    if not isinstance(decoded, Mapping):
        raise RuntimeError(f"{variant} teacher decoder must return a mapping")
    logits = decoded.get("action_logits")
    if logits is None:
        logits = decoded.get("action_logits_final")
    if not isinstance(logits, Tensor):
        raise RuntimeError(f"{variant} teacher decoder output is missing action logits")
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if tuple(logits.shape) != (1, action_dim):
        raise RuntimeError(
            f"{variant} teacher action logits must have shape [1,{action_dim}], "
            f"got {tuple(logits.shape)}"
        )
    return logits.to(reference)


def build_sparse_counterfactual_teacher(
    base_action_logits: Tensor,
    action_targets: Tensor,
    candidate_weight: Tensor,
    predicate_reliability: Tensor,
    base_predicate_overlap: Tensor,
    *,
    detail_field: Tensor,
    predicate_map: Tensor,
    action_contribution: Tensor,
    grid_hw: tuple[int, int] = (45, 80),
    utility_logit: Tensor | None = None,
    teacher_decoder: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Create selected/control variants from one encoded field.

    The same decoder is called once for selected deletion and once for its
    exact matched control. No DINO call or full field reconstruction occurs.
    """
    if teacher_decoder is None:
        raise RuntimeError("counterfactual teacher requires a selected/control decoder")
    plan = build_sparse_teacher_plan(
        base_action_logits,
        action_targets,
        candidate_weight,
        predicate_reliability,
        base_predicate_overlap,
    )
    if plan["factor_indices"].numel() == 0:
        # A strict teacher may abstain when no reliable factor/action proposal
        # exists. This must not halt the primary action/reason update or create
        # a synthetic fallback control.
        empty_long = torch.empty(0, dtype=torch.long, device=base_action_logits.device)
        empty_value = base_action_logits.new_empty((0,))
        for key in (
            "sample_indices",
            "action_indices",
            "factor_indices",
            "teacher_sample_indices",
            "teacher_action_indices",
            "teacher_factor_indices",
        ):
            plan[key] = empty_long
        for key in (
            "candidate_scores",
            "selected_candidate_weight",
            "selected_reliability",
            "selected_base_overlap",
            "selected_deletion_margin",
            "control_margin",
        ):
            plan[key] = empty_value
        plan.update(
            {
                "utility_teacher_target": None,
                "utility_teacher_prediction": None,
                "records": [],
                "selected_control_calls": 0,
                "available": False,
                "candidate_count": 0,
                "matched_control_count": 0,
                "unmatched_control_count": 0,
                "rejected_controls": [],
            }
        )
        return plan
    detail_field = _require_tensor(detail_field, "detail_field", 3)
    predicate_map = _require_tensor(predicate_map, "predicate_map", 3)
    action_contribution = _require_tensor(action_contribution, "action_contribution", 3)
    if detail_field.shape[0] != base_action_logits.shape[0]:
        raise ValueError("detail_field batch must match base_action_logits")
    if predicate_map.shape[0] != detail_field.shape[0]:
        raise ValueError("predicate_map batch must match detail_field")
    if action_contribution.shape[:2] != base_action_logits.shape:
        raise ValueError("action_contribution must have shape [B,A,N]")
    if predicate_map.shape[-1] != detail_field.shape[1]:
        raise ValueError("predicate_map patch count must match detail_field")
    if action_contribution.shape[-1] != detail_field.shape[1]:
        raise ValueError("action_contribution patch count must match detail_field")
    if utility_logit is not None:
        utility_logit = _require_tensor(utility_logit, "utility_logit", 3)
        expected = (
            base_action_logits.shape[0],
            base_action_logits.shape[1],
            predicate_map.shape[1],
        )
        if tuple(utility_logit.shape) != expected:
            raise ValueError(f"utility_logit must have shape {expected}")
    records: list[dict[str, Any]] = []
    teacher_targets: list[Tensor] = []
    teacher_predictions: list[Tensor] = []
    selected_margins: list[Tensor] = []
    control_margins: list[Tensor] = []
    accepted_positions: list[int] = []
    rejected_controls: list[dict[str, Any]] = []
    for position in range(int(plan["factor_indices"].numel())):
        sample = int(plan["sample_indices"][position].item())
        action = int(plan["action_indices"][position].item())
        factor = int(plan["factor_indices"][position].item())
        score = plan["candidate_scores"][position]
        selected = select_selected_patches(
            predicate_map[sample],
            torch.tensor([factor], device=predicate_map.device),
            candidate_scores=score.reshape(1),
        )
        try:
            control, metadata = select_matched_control_patches(
                detail_field[sample],
                selected,
                grid_hw=grid_hw,
                predicate_map=predicate_map[sample],
                action_contribution=action_contribution[sample, action],
            )
        except ValueError as error:
            # Strict matched controls are a validity condition, not a reason to
            # terminate the primary training update.  An unmatchable candidate
            # is excluded from this sparse teacher pass; it is never replaced
            # by a random or relaxed control.
            if "insufficient exact matched controls" not in str(error):
                raise
            rejected_controls.append(
                {
                    "sample_index": sample,
                    "action_index": action,
                    "factor_index": factor,
                    "reason": "insufficient_exact_matched_controls",
                }
            )
            continue
        if selected.numel() != control.numel() or bool(torch.isin(control, selected).any()):
            raise RuntimeError("counterfactual teacher produced self-selected or unequal control")
        field = detail_field[sample : sample + 1]
        with torch.no_grad():
            selected_decoded = teacher_decoder(
                field,
                selected,
                sample_index=sample,
                action_index=action,
                factor_index=factor,
                variant="selected",
            )
            control_decoded = teacher_decoder(
                field,
                control,
                sample_index=sample,
                action_index=action,
                factor_index=factor,
                variant="control",
            )
            selected_logits = _teacher_action_logits(
                selected_decoded,
                variant="selected",
                action_dim=base_action_logits.shape[1],
                reference=base_action_logits,
            )
            control_logits = _teacher_action_logits(
                control_decoded,
                variant="control",
                action_dim=base_action_logits.shape[1],
                reference=base_action_logits,
            )
            sign = action_targets[sample, action].float() * 2.0 - 1.0
            selected_margin = sign * selected_logits[0, action]
            control_margin = sign * control_logits[0, action]
            target = torch.sigmoid((control_margin - selected_margin) / 0.10)
        selected_margins.append(selected_margin.detach())
        control_margins.append(control_margin.detach())
        teacher_targets.append(target.detach())
        if utility_logit is not None:
            teacher_predictions.append(utility_logit[sample, action, factor])
        accepted_positions.append(position)
        record: dict[str, Any] = {
            "sample_index": sample,
            "action_index": action,
            "factor_index": factor,
            "selected_patches": selected,
            "control_patches": control,
            "control_metadata": metadata,
            "selected_deletion_margin": selected_margin.detach(),
            "control_margin": control_margin.detach(),
            "utility_teacher_target": target.detach(),
        }
        records.append(record)
    candidate_count = int(plan["factor_indices"].numel())
    if not accepted_positions:
        empty_long = plan["factor_indices"].new_empty((0,))
        empty_value = base_action_logits.new_empty((0,))
        for key in (
            "sample_indices",
            "action_indices",
            "factor_indices",
            "teacher_sample_indices",
            "teacher_action_indices",
            "teacher_factor_indices",
        ):
            plan[key] = empty_long
        for key in (
            "candidate_scores",
            "selected_candidate_weight",
            "selected_reliability",
            "selected_base_overlap",
            "selected_deletion_margin",
            "control_margin",
        ):
            plan[key] = empty_value
        plan["utility_teacher_target"] = None
        plan["utility_teacher_prediction"] = None
        plan["records"] = []
        plan["selected_control_calls"] = 0
        plan["available"] = False
        plan["candidate_count"] = candidate_count
        plan["matched_control_count"] = 0
        plan["unmatched_control_count"] = len(rejected_controls)
        plan["rejected_controls"] = rejected_controls
        return plan

    keep = torch.tensor(accepted_positions, device=plan["factor_indices"].device)
    for key in (
        "sample_indices",
        "action_indices",
        "factor_indices",
        "candidate_scores",
        "selected_candidate_weight",
        "selected_reliability",
        "selected_base_overlap",
        "teacher_sample_indices",
        "teacher_action_indices",
        "teacher_factor_indices",
    ):
        plan[key] = plan[key].index_select(0, keep)
    plan["selected_deletion_margin"] = torch.stack(selected_margins)
    plan["control_margin"] = torch.stack(control_margins)
    plan["utility_teacher_target"] = torch.stack(teacher_targets)
    plan["utility_teacher_prediction"] = (
        torch.stack(teacher_predictions) if teacher_predictions else None
    )
    plan["records"] = records
    plan["selected_control_calls"] = 2 * len(records)
    plan["available"] = True
    plan["candidate_count"] = candidate_count
    plan["matched_control_count"] = len(records)
    plan["unmatched_control_count"] = len(rejected_controls)
    plan["rejected_controls"] = rejected_controls
    return plan


SAVEUtilityPredictor = SAVEUtilityBridge
SAVEActionUtilityBridge = SAVEUtilityBridge
SAVEActionValidatedUtility = SAVEUtilityBridge
build_sparse_teacher = build_sparse_counterfactual_teacher


__all__ = [
    "SAVEActionUtilityBridge",
    "SAVEActionValidatedUtility",
    "SAVEUtilityBridge",
    "SAVEUtilityPredictor",
    "SAVE_UTILITY_MAX_SELECTED_PATCHES",
    "SAVE_UTILITY_MAX_TEACHER_PREDICATES",
    "SAVE_UTILITY_MAX_TEACHER_SAMPLES",
    "SAVE_UTILITY_MIN_SELECTED_PATCHES",
    "SAVE_UTILITY_RANK",
    "SAVE_UTILITY_TEACHER_INTERVAL_UPDATES",
    "build_sparse_counterfactual_teacher",
    "build_sparse_teacher",
    "build_sparse_teacher_plan",
    "select_matched_control_patches",
    "select_geometry_matched_control",
    "select_selected_patches",
    "select_sparse_teacher_targets",
    "select_teacher_predicates",
]
