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
