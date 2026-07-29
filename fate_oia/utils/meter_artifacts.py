from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from fate_oia.utils.tesa_contracts import patch_audit_contract_failures


def combined_file_hash(*paths: str | Path) -> str:
    """Hash path identity and bytes using the canonical readiness algorithm."""
    digest = hashlib.sha256()
    for value in paths:
        path = Path(value)
        digest.update(str(path).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def python_source_tree_hash(root: str | Path) -> str:
    """Hash every tracked Python source location used by METER readiness."""
    base = Path(root)
    digest = hashlib.sha256()
    for path in sorted(base.glob("fate_oia/**/*.py")):
        digest.update(str(path.relative_to(base)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            _json_safe(dict(value)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                _json_safe(dict(value)), sort_keys=True, allow_nan=False
            )
            + "\n"
        )


def save_meter_tensor(path: str | Path, value: torch.Tensor) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value.detach().cpu(), target)


def state_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("numpy") is not None:
        np.random.set_state(state["numpy"])
    if state.get("torch") is not None:
        torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    micro_step: int,
    optimizer_step: int,
    runtime_profile: Mapping[str, Any],
    meta_state: Mapping[str, Any],
    pu_state: Mapping[str, Any],
    calibration: Mapping[str, Any] | None,
    config_hash: str,
    source_hash: str,
    schema_hash: str,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "micro_step": int(micro_step),
        "optimizer_step": int(optimizer_step),
        "rng_state": capture_rng_state(),
        "runtime_profile": dict(runtime_profile),
        "meta_state": dict(meta_state),
        "pu_state": dict(pu_state),
        "calibration": dict(calibration or {}),
        "config_hash": str(config_hash),
        "source_hash": str(source_hash),
        "schema_hash": str(schema_hash),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    expected_config_hash: str | None = None,
    expected_source_hash: str | None = None,
    expected_schema_hash: str | None = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key, expected in (
        ("config_hash", expected_config_hash),
        ("source_hash", expected_source_hash),
        ("schema_hash", expected_schema_hash),
    ):
        if expected is not None and payload.get(key) != expected:
            raise ValueError(f"Checkpoint {key} mismatch")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if payload.get("rng_state"):
        restore_rng_state(payload["rng_state"])
    return payload


def save_epoch_artifacts(
    root: str | Path,
    epoch: int,
    *,
    metrics_raw: Mapping[str, Any],
    metrics_deploy: Mapping[str, Any],
    branch_metrics: Mapping[str, Any],
    logits: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    diagnostics: Mapping[str, Any],
    file_names: list[str] | None = None,
) -> Path:
    directory = Path(root) / f"epoch_{int(epoch):03d}"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "metrics_raw.json", metrics_raw)
    write_json(directory / "metrics_deploy.json", metrics_deploy)
    write_json(directory / "branch_metrics.json", branch_metrics)
    for name, value in logits.items():
        save_meter_tensor(directory / f"logits_{name}.pt", value)
    for name, value in labels.items():
        save_meter_tensor(directory / f"labels_{name}.pt", value)
    if file_names is not None:
        write_json(directory / "file_names_test.json", {"file_names": list(file_names)})
    for name, value in diagnostics.items():
        if name.endswith(".jsonl"):
            if isinstance(value, list):
                for row in value:
                    append_jsonl(
                        directory / name,
                        row if isinstance(row, Mapping) else {"value": row},
                    )
            else:
                append_jsonl(directory / name, value if isinstance(value, Mapping) else {"value": value})
        else:
            write_json(directory / (name if name.endswith(".json") else f"{name}.json"), value if isinstance(value, Mapping) else {"value": value})
    return directory


def validate_epoch_artifacts(directory: str | Path) -> list[str]:
    root = Path(directory)
    required = [
        "metrics_raw.json", "metrics_deploy.json", "branch_metrics.json",
        "typed_evidence.json", "pu_stats.json", "calibration.json", "runtime.json",
        "file_names_test.json",
        "logits_action_final_raw_test.pt", "logits_reason_final_raw_test.pt",
        "logits_action_visual_test.pt", "logits_reason_global_test.pt",
        "labels_action_test.pt", "labels_reason_test.pt",
    ]
    failures = [name for name in required if not (root / name).exists()]
    if failures:
        return failures
    try:
        file_names = json.loads(
            (root / "file_names_test.json").read_text(encoding="utf-8")
        )["file_names"]
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        failures.append("file_names_test.json:schema")
        return failures
    expected_rows = len(file_names)
    tensor_shapes = {
        "logits_action_final_raw_test.pt": (expected_rows, 4),
        "logits_reason_final_raw_test.pt": (expected_rows, 21),
        "logits_action_visual_test.pt": (expected_rows, 4),
        "logits_reason_global_test.pt": (expected_rows, 21),
        "labels_action_test.pt": (expected_rows, 4),
        "labels_reason_test.pt": (expected_rows, 21),
    }
    for name, expected_shape in tensor_shapes.items():
        try:
            value = torch.load(root / name, map_location="cpu", weights_only=False)
        except Exception:
            failures.append(f"{name}:unreadable")
            continue
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shape:
            failures.append(f"{name}:shape")
            continue
        if not bool(torch.isfinite(value).all()):
            failures.append(f"{name}:non_finite")
    payloads: dict[str, dict[str, Any]] = {}
    for name in (
        "metrics_raw.json",
        "metrics_deploy.json",
        "branch_metrics.json",
        "typed_evidence.json",
        "pu_stats.json",
        "calibration.json",
        "runtime.json",
    ):
        try:
            payload = json.loads((root / name).read_text(encoding="utf-8"))
        except (ValueError, json.JSONDecodeError):
            failures.append(f"{name}:invalid_json")
            continue
        if not isinstance(payload, dict):
            failures.append(f"{name}:schema")
            continue
        payloads[name] = payload

    typed = payloads.get("typed_evidence.json", {})
    typed_lengths = {
        "state_confusion_matrix": 21,
        "source_coverage": 21,
        "same_type_margin": 21,
        "mirror_equivariance": 21,
        "identity_target_delta": 4,
        "identity_wrong_delta": 4,
        "factor_off_delta": 4,
        "state_off_delta": 4,
        "cross_sample_swap_effect": 4,
    }
    typed_valid = all(
        isinstance(typed.get(key), list) and len(typed[key]) == length
        for key, length in typed_lengths.items()
    )
    confusion = typed.get("state_confusion_matrix", [])
    typed_valid = typed_valid and all(
        isinstance(matrix, list)
        and len(matrix) == 3
        and all(
            isinstance(row, list)
            and len(row) == 3
            and all(
                isinstance(value, int) and value >= 0 for value in row
            )
            for row in matrix
        )
        for matrix in confusion
    )
    identity_matrix = typed.get("identity_ap_delta_matrix")
    typed_valid = (
        typed_valid
        and isinstance(identity_matrix, list)
        and len(identity_matrix) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in identity_matrix)
        and all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for row in identity_matrix or []
            for value in row
        )
    )
    numeric_vectors = (
        "identity_target_delta",
        "identity_wrong_delta",
        "factor_off_delta",
        "state_off_delta",
        "cross_sample_swap_effect",
    )
    typed_valid = typed_valid and all(
        all(
            isinstance(value, (int, float)) and math.isfinite(float(value))
            for value in typed[key]
        )
        for key in numeric_vectors
        if isinstance(typed.get(key), list)
    )
    if typed_valid:
        expected_target = [
            float(identity_matrix[action][action]) for action in range(4)
        ]
        expected_wrong = [
            sum(
                abs(float(identity_matrix[target][action]))
                for action in range(4)
                if action != target
            )
            / 3.0
            for target in range(4)
        ]
        typed_valid = all(
            abs(float(actual) - expected) < 1e-8
            for actual, expected in zip(
                typed["identity_target_delta"], expected_target
            )
        ) and all(
            abs(float(actual) - expected) < 1e-8
            for actual, expected in zip(
                typed["identity_wrong_delta"], expected_wrong
            )
        )
    train_audit = typed.get("train_audit", {})
    patch_audit = typed.get("patch_audit", {})
    patch_contract_failures = patch_audit_contract_failures(patch_audit)
    typed_valid = (
        typed_valid
        and isinstance(train_audit, dict)
        and isinstance(train_audit.get("per_factor"), list)
        and len(train_audit["per_factor"]) == 21
        and isinstance(patch_audit, dict)
        and isinstance(patch_audit.get("unique_sample_count"), int)
        and isinstance(patch_audit.get("action_coverage"), list)
        and isinstance(patch_audit.get("factor_coverage"), list)
        and all(
            isinstance(value, int) and 0 <= value < 4
            for value in patch_audit.get("action_coverage", [])
        )
        and all(
            isinstance(value, int) and 0 <= value < 21
            for value in patch_audit.get("factor_coverage", [])
        )
    )
    # New artifacts carry separated coverage/CI fields. Historical pilot
    # artifacts remain readable so the evaluator can diagnose them honestly.
    if any(name in patch_audit for name in (
        "eligible_factor_coverage",
        "requested_factor_coverage",
        "executed_factor_coverage",
        "model_top_factor_coverage",
        "selected_minus_control_ci",
    )):
        typed_valid = typed_valid and not patch_contract_failures
    if not typed_valid:
        failures.append("typed_evidence.json:mechanism_schema")

    calibration = payloads.get("calibration.json", {})
    calibration_required = {
        "theta",
        "temperature",
        "strategy",
        "accepted",
        "fallback_reason",
        "fit_split",
        "representation_updated",
        "train_calib_raw_joint",
        "train_calib_deploy_joint",
    }
    if (
        not calibration_required.issubset(calibration)
        or not isinstance(calibration.get("theta"), list)
        or len(calibration.get("theta", [])) != 25
        or
        calibration.get("fit_split") != "train_calib"
        or calibration.get("representation_updated") is not False
    ):
        failures.append("calibration.json:train_calib_schema")

    runtime = payloads.get("runtime.json", {})
    runtime_required = {
        "epoch",
        "train_rows",
        "mean_data_time",
        "mean_dino_time",
        "peak_reserved_gb",
        "eval_mode_time",
        "dino_call_count",
    }
    if (
        not runtime_required.issubset(runtime)
        or
        not isinstance(runtime.get("dino_call_count"), dict)
        or not isinstance(runtime.get("eval_mode_time"), dict)
        or not isinstance(runtime.get("peak_reserved_gb"), (int, float))
        or not all(
            isinstance(runtime.get(key), (int, float))
            and math.isfinite(float(runtime[key]))
            for key in (
                "mean_data_time",
                "mean_dino_time",
                "peak_reserved_gb",
            )
        )
    ):
        failures.append("runtime.json:profile_schema")
    return failures
