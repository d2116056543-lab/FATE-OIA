from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor


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
    records: list[dict[str, Any]] = field(default_factory=list)
    _unique: set[str] = field(default_factory=set)

    @property
    def unique_count(self) -> int:
        return len(self._unique)

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


@torch.no_grad()
def run_stratified_patch_audit(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    *,
    progress: float,
    max_unique: int = 128,
    patches_per_factor: int = 12,
) -> dict[str, Any]:
    """Audit every eligible target for unique samples without re-running DINO."""
    queue = StratifiedPatchAudit(max_unique=max_unique)
    records: list[dict[str, Any]] = []
    action_coverage: set[int] = set()
    factor_coverage: set[int] = set()
    for batch in loader:
        if queue.unique_count >= max_unique:
            break
        images = batch["image"].to(device, non_blocking=True)
        field = model.encode_images(images)
        clean = model.decode_from_field(field, progress=progress)
        for sample, sample_id in enumerate(batch["file_name"]):
            if queue.unique_count >= max_unique:
                break
            positive_actions = torch.where(batch["action"][sample] > 0.5)[0].tolist()
            if not positive_actions:
                continue
            contributions = clean["action_factor_contributions"][sample]
            allowed = clean["factor_action_ownership"].to(contributions) > 0
            factors = sorted(
                {
                    int(
                        contributions[action]
                        .abs()
                        .masked_fill(~allowed, -1)
                        .argmax()
                    )
                    for action in positive_actions
                }
            )
            queue.add(
                str(sample_id),
                action_ids=[int(value) for value in positive_actions],
                factor_ids=factors,
            )
            for action in positive_actions:
                sign = 1.0
                for factor in factors:
                    anchor = clean["factor_anchor_map"][sample, factor]
                    count = min(int(patches_per_factor), anchor.numel() // 2)
                    selected = torch.topk(anchor, k=count).indices
                    peak = int(anchor.argmax())
                    width = 80
                    sector = min((peak % width) // max(width // 3, 1), 2)
                    columns = torch.arange(anchor.numel(), device=anchor.device) % width
                    candidate = (columns // max(width // 3, 1)).clamp_max(2) == sector
                    candidate[selected] = False
                    control_score = anchor.masked_fill(~candidate, float("inf"))
                    if int(candidate.sum()) < count:
                        candidate = torch.ones_like(candidate)
                        candidate[selected] = False
                        control_score = anchor.masked_fill(~candidate, float("inf"))
                    control = torch.topk(control_score, k=count, largest=False).indices
                    selected_output = model.decode_from_field(
                        _replace_patches(field, sample, selected), progress=progress
                    )
                    control_output = model.decode_from_field(
                        _replace_patches(field, sample, control), progress=progress
                    )
                    clean_logit = clean["action_logits_final"][sample, action]
                    selected_effect = sign * (
                        clean_logit - selected_output["action_logits_final"][0, action]
                    )
                    control_effect = sign * (
                        clean_logit - control_output["action_logits_final"][0, action]
                    )
                    records.append(
                        {
                            "sample_id": str(sample_id),
                            "action_id": int(action),
                            "factor_id": int(factor),
                            "selected_effect": float(selected_effect),
                            "control_effect": float(control_effect),
                            "selected_minus_control": float(
                                selected_effect - control_effect
                            ),
                        }
                    )
                    action_coverage.add(int(action))
                    factor_coverage.add(int(factor))
        del clean, field, images
    selected = [row["selected_effect"] for row in records]
    control = [row["control_effect"] for row in records]
    gaps = [row["selected_minus_control"] for row in records]
    return {
        "available": bool(records),
        "unique_sample_count": queue.unique_count,
        "cumulative_unique_count": queue.unique_count,
        "action_coverage": sorted(action_coverage),
        "factor_coverage": sorted(factor_coverage),
        "selected_effect_mean": sum(selected) / max(len(selected), 1),
        "control_effect_mean": sum(control) / max(len(control), 1),
        "selected_minus_control_mean": sum(gaps) / max(len(gaps), 1),
        "records": records,
    }
