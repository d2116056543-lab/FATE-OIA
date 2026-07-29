from __future__ import annotations

import gc
import random
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

from fate_oia.utils.tesa_contracts import PATCH_AUDIT_COVERAGE_FIELDS


SEQUENTIAL_EVAL_MODES = (
    "main",
    "visual",
    "factor_off",
    "reason_global",
    "reason_correction_off",
    "counterfactual",
)

REQUIRED_TESA_ARTIFACT_FIELDS = {
    "action_visual_logits", "action_final_logits", "per_action_f1",
    "per_action_ap", "per_action_auc", "action_correction_rms",
    "action_factor_contributions", "action_factor_weights",
    "identity_target_delta", "identity_wrong_delta", "factor_off_delta",
    "state_off_delta", "reason_global_logits", "reason_final_logits",
    "per_reason_f1", "per_reason_ap", "per_reason_auc",
    "reason_correction_rms", "groundable_metrics", "latent_metrics",
    "pu_active_labels", "pu_lambda", "anchor_entropy", "null_mass",
    "observability", "state_entropy", "state_confusion_matrix",
    "source_coverage", "same_type_margin", "mirror_equivariance",
    "analytic_coverage", "correct_factor_effect", "wrong_factor_effect",
    "cross_sample_swap_effect", "patch_selected_effect",
    "patch_control_effect", "unique_sample_count", "cumulative_unique_count",
    "action_coverage", "factor_coverage", "temperature", "threshold_vector",
    "train_calib_raw_joint", "train_calib_deploy_joint",
    "threshold_shrinkage_state", "fallback_reason", "data_time", "dino_time",
    "foundation_time", "factor_time", "action_time", "reason_time",
    "backward_time", "eval_mode_time", "allocated_gb", "reserved_gb",
    "dino_call_count",
    *PATCH_AUDIT_COVERAGE_FIELDS,
}


def schema_token_mismatch(token: Tensor) -> Tensor:
    return torch.roll(token, shifts=-1, dims=1)


def cross_sample_same_factor_swap(token: Tensor) -> Tensor:
    return torch.roll(token, shifts=1, dims=0)


def corrupt_state_keep_anchor(
    anchor: Tensor, state_probability: Tensor
) -> tuple[Tensor, Tensor]:
    return anchor.clone(), torch.roll(state_probability, shifts=1, dims=-1)


def mechanism_ramps(optimizer_step: int, total_updates: int) -> tuple[float, float]:
    r5 = min(max(optimizer_step / max(0.05 * total_updates, 1.0), 0.0), 1.0)
    r10 = min(max(optimizer_step / max(0.10 * total_updates, 1.0), 0.0), 1.0)
    return 0.25 + 0.75 * r5, r10


@dataclass
class StratifiedPatchAudit:
    max_unique: int = 128
    previous_ids: set[str] = field(default_factory=set)
    records: list[dict[str, Any]] = field(default_factory=list)
    _unique: set[str] = field(default_factory=set)

    @property
    def unique_count(self) -> int:
        return len(self._unique)

    @property
    def cumulative_unique_count(self) -> int:
        return len(self.previous_ids | self._unique)

    def add(
        self, sample_id: str, *, action_ids: list[int], factor_ids: list[int]
    ) -> None:
        if sample_id not in self._unique and len(self._unique) >= self.max_unique:
            return
        self._unique.add(sample_id)
        for action_id in action_ids:
            for factor_id in factor_ids:
                self.records.append(
                    {
                        "sample_id": sample_id,
                        "action_id": int(action_id),
                        "factor_id": int(factor_id),
                    }
                )


def sequential_evaluate(
    evaluator: Callable[[str], dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for mode in SEQUENTIAL_EVAL_MODES:
        result[mode] = evaluator(mode)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    return result


def _replace_patches(field: dict[str, Any], sample: int, indices: Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in field.items():
        if not isinstance(value, Tensor) or value.shape[0] != field["patch_tokens_by_layer"].shape[0]:
            result[key] = value
            continue
        selected = value[sample : sample + 1].clone()
        if key == "patch_tokens_by_layer":
            keep = torch.ones(selected.shape[2], dtype=torch.bool, device=selected.device)
            keep[indices] = False
            replacement = selected[:, :, keep].mean(2, keepdim=True)
            selected[:, :, indices] = replacement
        result[key] = selected
    return result


def _patch_geometry(index: Tensor, grid_hw: tuple[int, int]) -> tuple[int, int]:
    height, width = grid_hw
    rows = torch.div(index, width, rounding_mode="floor")
    columns = index.remainder(width)
    side = int((columns.float().mean() * 3.0 / width).floor().clamp(0, 2).item())
    depth = int((rows.float().mean() * 5.0 / height).floor().clamp(0, 4).item())
    return side, depth


def source_eligible_factor_mask(
    factor_source_weight: Tensor,
    factor_anchor_valid: Tensor,
    factor_groundable_mask: Tensor,
) -> Tensor:
    """Return source eligibility without consulting any model prediction."""
    return (
        (factor_source_weight > 0)
        & factor_anchor_valid.bool()
        & factor_groundable_mask.bool()
    )


def select_geometry_matched_control(
    anchor: Tensor,
    selected: Tensor,
    *,
    grid_hw: tuple[int, int],
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, dict[str, Any]]:
    """Pick a disjoint control matching count, lateral side, depth, and validity."""
    height, width = grid_hw
    if anchor.ndim != 1 or anchor.numel() != height * width:
        raise ValueError("anchor must be a flattened grid")
    valid = (
        valid_mask.to(device=anchor.device, dtype=torch.bool)
        if valid_mask is not None
        else torch.ones_like(anchor, dtype=torch.bool)
    )
    if valid.shape != anchor.shape:
        raise ValueError("valid_mask must match anchor")
    selected_side, selected_depth = _patch_geometry(selected, grid_hw)
    positions = torch.arange(anchor.numel(), device=anchor.device)
    rows = torch.div(positions, width, rounding_mode="floor")
    columns = positions.remainder(width)
    sides = (columns * 3 // width).clamp_max(2)
    depths = (rows * 5 // height).clamp_max(4)
    candidate = valid & (sides == selected_side) & (depths == selected_depth)
    candidate[selected] = False
    count = selected.numel()
    if int(candidate.sum()) < count:
        raise ValueError("insufficient geometry-matched valid control patches")
    control = torch.topk(
        anchor.masked_fill(~candidate, float("inf")), k=count, largest=False
    ).indices
    control_side, control_depth = _patch_geometry(control, grid_hw)
    return control, {
        "selected_count": int(count),
        "control_count": int(control.numel()),
        "selected_side": selected_side,
        "control_side": control_side,
        "selected_depth_bin": selected_depth,
        "control_depth_bin": control_depth,
        "control_valid_fraction": float(valid[control].float().mean().item()),
        "overlap_count": int(torch.isin(control, selected).sum().item()),
    }


def _bootstrap_mean_ci(
    values: list[float], *, samples: int, seed: int
) -> dict[str, float | int]:
    if not values:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "n_bootstrap": int(samples)}
    generator = random.Random(seed)
    means = [
        sum(values[generator.randrange(len(values))] for _ in values) / len(values)
        for _ in range(max(1, int(samples)))
    ]
    means.sort()
    return {
        "mean": sum(values) / len(values),
        "low": means[int(0.025 * (len(means) - 1))],
        "high": means[int(0.975 * (len(means) - 1))],
        "n_bootstrap": len(means),
    }


def _clustered_gap_ci(
    records: list[dict[str, Any]], *, samples: int, seed: int
) -> dict[str, float | int]:
    by_sample: dict[str, list[float]] = {}
    for row in records:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id:
            raise ValueError("Patch deletion records require sample_id")
        by_sample.setdefault(sample_id, []).append(
            float(row["selected_minus_control"])
        )
    sample_means = [
        sum(values) / len(values) for values in by_sample.values()
    ]
    result = _bootstrap_mean_ci(sample_means, samples=samples, seed=seed)
    result["cluster_count"] = len(sample_means)
    return result


def build_patch_audit_summary(
    records: list[dict[str, Any]],
    *,
    sample_ids: set[str],
    cumulative_sample_ids: set[str],
    eligible_factor_ids: set[int],
    requested_factor_ids: set[int],
    model_top_factor_ids: set[int],
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 20260729,
) -> dict[str, Any]:
    """Keep source coverage separate from model-top deletion faithfulness."""
    selected = [float(row["selected_effect"]) for row in records]
    control = [float(row["control_effect"]) for row in records]
    gaps = [float(row["selected_minus_control"]) for row in records]
    executed = {int(row["factor_id"]) for row in records}
    actions = {int(row["action_id"]) for row in records}
    return {
        "available": bool(records),
        "unique_sample_count": len(sample_ids),
        "cumulative_unique_count": len(cumulative_sample_ids),
        "sample_ids": sorted(cumulative_sample_ids),
        "action_coverage": sorted(actions),
        "eligible_factor_coverage": sorted(eligible_factor_ids),
        "requested_factor_coverage": sorted(requested_factor_ids),
        "executed_factor_coverage": sorted(executed),
        "model_top_factor_coverage": sorted(model_top_factor_ids),
        "factor_coverage": sorted(executed),
        "selected_effect_mean": sum(selected) / max(len(selected), 1),
        "control_effect_mean": sum(control) / max(len(control), 1),
        "selected_minus_control_mean": sum(gaps) / max(len(gaps), 1),
        "selected_minus_control_ci": _clustered_gap_ci(
            records, samples=bootstrap_samples, seed=bootstrap_seed
        ),
        "selected_positive_rate": sum(value > 0.0 for value in selected) / max(len(selected), 1),
        "records": records,
    }


@torch.no_grad()
def run_stratified_patch_audit(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    progress: float,
    max_unique: int = 128,
    patches_per_factor: int = 12,
    factors_per_action: int = 2,
    previous_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Audit every eligible target for unique samples without re-running DINO."""
    queue = StratifiedPatchAudit(
        max_unique=max_unique, previous_ids=set(previous_sample_ids or ())
    )
    records: list[dict[str, Any]] = []
    action_coverage: set[int] = set()
    eligible_factor_coverage: set[int] = set()
    requested_factor_coverage: set[int] = set()
    model_top_factor_coverage: set[int] = set()
    missing_source_count = 0
    for batch in loader:
        if queue.unique_count >= max_unique:
            break
        images = batch["image"].to(device, non_blocking=True)
        field = model.encode_images(images)
        clean = model.decode_from_field(field, progress=progress)
        for sample, sample_id in enumerate(batch["file_name"]):
            if queue.unique_count >= max_unique:
                break
            action_logits = clean["action_logits_final"][sample]
            predicted_actions = torch.where(torch.sigmoid(action_logits) > 0.5)[
                0
            ]
            if predicted_actions.numel() == 0:
                predicted_actions = action_logits.argmax().view(1)
            if predicted_actions.numel() > 2:
                predicted_actions = torch.topk(action_logits, k=2).indices
            selected_actions = [int(value) for value in predicted_actions.tolist()]
            contributions = clean["action_factor_contributions"][sample]
            allowed = clean["factor_action_ownership"].to(contributions) > 0
            groundable = clean["factor_groundable_mask"].to(contributions) > 0.5
            grounding = batch.get("meter_grounding", {})
            if isinstance(grounding, dict) and {
                "factor_source_weight",
                "factor_anchor_valid",
            }.issubset(grounding):
                source_eligible = source_eligible_factor_mask(
                    grounding["factor_source_weight"][sample].to(contributions),
                    grounding["factor_anchor_valid"][sample].to(contributions),
                    groundable,
                )
            else:
                missing_source_count += 1
                continue
            eligible_factor_coverage.update(
                int(value) for value in torch.where(source_eligible)[0].tolist()
            )
            factors_by_action: dict[int, list[int]] = {}
            for action in selected_actions:
                eligible = source_eligible & allowed
                top_count = min(int(factors_per_action), int(eligible.sum()))
                if top_count == 0:
                    factors_by_action[int(action)] = []
                    continue
                top_candidates = torch.topk(
                    contributions[action].abs().masked_fill(~eligible, -1),
                    k=top_count,
                ).indices.tolist()
                model_top_factor_coverage.update(
                    int(value) for value in top_candidates
                )
                factors_by_action[int(action)] = [
                    int(factor) for factor in top_candidates
                ]
                for factor in top_candidates:
                    requested_factor_coverage.add(int(factor))
            factors = sorted(
                {int(factor) for values in factors_by_action.values() for factor in values}
            )
            queue.add(
                str(sample_id),
                action_ids=selected_actions,
                factor_ids=factors,
            )
            for action in selected_actions:
                for factor in factors_by_action[int(action)]:
                    contribution = contributions[action, factor]
                    anchor = clean["factor_anchor_map"][sample, factor]
                    count = min(int(patches_per_factor), anchor.numel() // 2)
                    selected = torch.topk(anchor, k=count).indices
                    valid_mask = field.get("valid_patch_mask")
                    if isinstance(valid_mask, Tensor):
                        valid_mask = valid_mask[sample]
                    try:
                        control, control_match = select_geometry_matched_control(
                            anchor,
                            selected,
                            grid_hw=(45, 80),
                            valid_mask=valid_mask,
                        )
                    except ValueError:
                        continue
                    selected_output = model.decode_from_field(
                        _replace_patches(field, sample, selected), progress=progress
                    )
                    control_output = model.decode_from_field(
                        _replace_patches(field, sample, control), progress=progress
                    )
                    clean_logit = clean["action_logits_final"][sample, action]
                    selected_effect = (
                        clean_logit - selected_output["action_logits_final"][0, action]
                    )
                    control_effect = (
                        clean_logit - control_output["action_logits_final"][0, action]
                    )
                    records.append(
                        {
                            "sample_id": str(sample_id),
                            "action_id": int(action),
                            "factor_id": int(factor),
                            "selection_mode": "model_top_predicted_action",
                            "clean_action_logit": float(clean_logit),
                            "factor_contribution": float(contribution),
                            "selected_effect": float(selected_effect),
                            "control_effect": float(control_effect),
                            "selected_minus_control": float(
                                selected_effect - control_effect
                            ),
                            "control_match": control_match,
                        }
                    )
                    action_coverage.add(int(action))
        del clean, field, images
    summary = build_patch_audit_summary(
        records,
        sample_ids=queue._unique,
        cumulative_sample_ids=queue.previous_ids | queue._unique,
        eligible_factor_ids=eligible_factor_coverage,
        requested_factor_ids=requested_factor_coverage,
        model_top_factor_ids=model_top_factor_coverage,
    )
    summary["action_coverage"] = sorted(action_coverage)
    summary["missing_source_count"] = missing_source_count
    return summary
