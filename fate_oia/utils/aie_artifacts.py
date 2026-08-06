from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import random
import numpy as np


def json_safe(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().tolist() if value.ndim else value.detach().cpu().item()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(json_safe(value), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, value: Any) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(value), ensure_ascii=False) + "\n")


REQUIRED_STEP_FILES = ("loss_components.jsonl", "owner_gradients.jsonl", "runtime_components.jsonl", "evidence_components.jsonl")
REQUIRED_RUN_FILES = (
    "config_resolved.yaml", "run_manifest.json", "source_contract.json", "owner_map.json", "split_manifest.json",
    "train_calib_ids.json", "train_audit_ids.json", "checkpoint_latest.pth",
)
REQUIRED_BOUND_RUN_FILES = ("AIE_IMPLEMENTATION_REVIEW.json", "AIE_RUNTIME_PROFILE.json")
REQUIRED_EPOCH_FILES = (
    "metrics_summary.json", "branch_metrics.json", "calibration.json", "predicate_metrics.json",
    "naming_metrics.json", "probe_metrics.json", "counterfactual_metrics.json", "owner_metrics.json", "runtime_metrics.json",
    "metrics_summary.jsonl", "branch_metrics.jsonl", "per_label_action_metrics.json", "per_label_reason_metrics.json",
    "calibration_diagnostics.jsonl", "predicate_metrics.jsonl", "predicate_grounding_metrics.jsonl",
    "naming_metrics.jsonl", "probe_metrics.jsonl", "counterfactual_metrics.jsonl", "owner_gradient_metrics.jsonl", "runtime_epoch_metrics.jsonl",
    "train_audit_metrics.json", "action_logits_primary_test.pt", "action_logits_final_test.pt",
    "reason_logits_primary_test.pt", "reason_logits_final_test.pt", "labels_action_test.pt", "labels_reason_test.pt",
    "file_names_test.json", "audit_128_ablation_logits.pt", "audit_128_full_tensors.pt", "audit_128_counterfactual_cases.json",
)

REQUIRED_LOSS_FIELDS = (
    "epoch", "micro_step", "optimizer_update", "learning_rates", "loss_total",
    "primary_action_logit_rms", "final_action_logit_rms", "action_delta_rms",
    "primary_reason_logit_rms", "final_reason_logit_rms", "reason_delta_rms",
    "raw_contribution_std", "bounded_contribution_std", "probe_map_entropy", "probe_pairwise_overlap",
    "primary_grad_raw", "primary_grad_capped", "action_evidence_grad", "action_contribution_grad", "reason_private_grad", "dino_grad",
    "data_time", "dino_time", "primary_time", "evidence_global_time", "evidence_local_time", "reason_reread_time", "backward_time",
)


def validate_artifact_files(root: str | Path, names: tuple[str, ...]) -> list[str]:
    base = Path(root)
    return [name for name in names if not (base / name).exists() or (base / name).stat().st_size == 0]


def validate_run_artifacts(root: str | Path) -> list[str]:
    base = Path(root)
    failures = [f"missing:{name}" for name in validate_artifact_files(base, REQUIRED_RUN_FILES + REQUIRED_STEP_FILES)]
    manifest_path = base / "run_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("run_kind") in {"pilot", "full"}:
                failures.extend(f"missing:{name}" for name in validate_artifact_files(base, REQUIRED_BOUND_RUN_FILES))
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(f"schema:run_manifest_invalid:{exc}")
    loss_path = base / "loss_components.jsonl"
    if loss_path.exists():
        try:
            rows = [json.loads(line) for line in loss_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                failures.append("schema:loss_components_empty")
            else:
                for row_index, row in enumerate(rows):
                    missing = [field for field in REQUIRED_LOSS_FIELDS if field not in row]
                    failures.extend(f"schema:loss_components:{row_index}:{field}" for field in missing)
        except (json.JSONDecodeError, OSError) as exc:
            failures.append(f"schema:loss_components_invalid:{exc}")
    epoch_dirs = sorted(base.glob("epoch_*"))
    if not epoch_dirs:
        failures.append("missing:epoch_directory")
    for epoch_dir in epoch_dirs:
        failures.extend(f"missing:{epoch_dir.name}/{name}" for name in validate_artifact_files(epoch_dir, REQUIRED_EPOCH_FILES))
    return failures


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
