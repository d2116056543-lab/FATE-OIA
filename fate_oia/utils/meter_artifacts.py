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


HECA_SIDECAR_FILES = {
    "ontology_manifest": "heca_ontology_manifest.json",
    "tau_stats": "heca_tau_stats.json",
    "gradient_ownership": "heca_gradient_ownership.jsonl",
    "loss_wiring": "heca_loss_wiring.json",
    "component_call_counters": "heca_component_call_counters.json",
    "contribution_conservation": "heca_contribution_conservation.jsonl",
    "schedule_state": "heca_schedule_state.json",
    "ablation_manifest": "heca_ablation_manifest.json",
}
HECA_GATE_NAMES = tuple("ABCDEFG")
HECA_CHEAP_MODE_NAMES = (
    "factor_off",
    "state_uniform",
    "reason_correction_off",
)
HECA_PILOT_EVIDENCE_MANIFEST = "heca_pilot_evidence_manifest.json"
HECA_PILOT_INPUT_FILES = (
    "heca_implementation_audit_input.json",
    "heca_ontology_manifest_input.json",
    "heca_tau_stats_input.json",
)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]] | None:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return rows if rows and all(isinstance(row, dict) for row in rows) else None


def write_heca_artifact_sidecar(
    directory: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write the strict HECA sidecar without inventing missing evidence."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    required = set(HECA_SIDECAR_FILES) | {"gates"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"HECA sidecar missing payloads: {', '.join(missing)}")
    for key, name in HECA_SIDECAR_FILES.items():
        value = payload[key]
        if name.endswith(".jsonl"):
            if not isinstance(value, list):
                raise ValueError(f"HECA sidecar {key} must be a JSONL row list")
            target = root / name
            target.unlink(missing_ok=True)
            for row in value:
                if not isinstance(row, Mapping):
                    raise ValueError(f"HECA sidecar {key} contains a non-object row")
                append_jsonl(target, row)
        else:
            if not isinstance(value, Mapping):
                raise ValueError(f"HECA sidecar {key} must be an object")
            write_json(root / name, value)
    gates = payload["gates"]
    if not isinstance(gates, Mapping):
        raise ValueError("HECA sidecar gates must be an object")
    for letter in HECA_GATE_NAMES:
        gate = gates.get(letter)
        if not isinstance(gate, Mapping):
            raise ValueError(f"HECA sidecar missing Gate {letter}")
        write_json(root / f"HECA_GATE_{letter}.json", gate)
    return root


def validate_heca_artifact_sidecar(directory: str | Path) -> list[str]:
    """Reject incomplete HECA artifacts instead of silently weakening audit."""
    root = Path(directory)
    failures: list[str] = []
    for name in HECA_SIDECAR_FILES.values():
        if not (root / name).exists():
            failures.append(f"{name}:missing")
    for letter in HECA_GATE_NAMES:
        name = f"HECA_GATE_{letter}.json"
        if not (root / name).exists():
            failures.append(f"{name}:missing")
    if failures:
        return failures

    ontology = _read_json(root / HECA_SIDECAR_FILES["ontology_manifest"])
    if not (
        ontology
        and isinstance(ontology.get("schema_version"), int)
        and ontology.get("factor_count") == 21
        and isinstance(ontology.get("state_count"), int)
        and ontology["state_count"] >= 1
        and isinstance(ontology.get("sha256"), str)
        and ontology["sha256"]
    ):
        failures.append("heca_ontology_manifest.json:schema")

    tau = _read_json(root / HECA_SIDECAR_FILES["tau_stats"])
    tau_values = tau.get("tau") if tau else None
    if not (
        tau
        and tau.get("source_split") == "train_main"
        and _is_finite_number(tau.get("alpha"))
        and len(tau_values or []) == 21
        and all(_is_finite_number(value) and 0.05 <= float(value) <= 0.95 for value in tau_values)
    ):
        failures.append("heca_tau_stats.json:schema")

    gradient_rows = _read_jsonl(root / HECA_SIDECAR_FILES["gradient_ownership"])
    gradient_fields = {
        "optimizer_step",
        "action_to_anchor_query",
        "action_to_state_bridge_ratio",
        "reason_to_action_credit",
        "measurement_to_foundation",
    }
    if not gradient_rows or not all(
        gradient_fields.issubset(row)
        and all(_is_finite_number(row[field]) for field in gradient_fields)
        for row in gradient_rows
    ):
        failures.append("heca_gradient_ownership.jsonl:schema")

    wiring = _read_json(root / HECA_SIDECAR_FILES["loss_wiring"])
    registry = wiring.get("registry") if wiring else None
    counts = wiring.get("counts") if wiring else None
    if not (
        wiring
        and isinstance(registry, list)
        and registry
        and all(isinstance(name, str) for name in registry)
        and isinstance(counts, dict)
        and all(counts.get(name) == 1 for name in registry)
        and wiring.get("duplicates") == []
        and wiring.get("pass") is True
    ):
        failures.append("heca_loss_wiring.json:schema")

    calls = _read_json(root / HECA_SIDECAR_FILES["component_call_counters"])
    components = calls.get("components") if calls else None
    required_components = {
        "dino_encode",
        "typed_measurement",
        "action_credit",
        "reason_correction",
    }
    if not (
        calls
        and calls.get("one_dino_encode_per_batch") is True
        and isinstance(components, dict)
        and all(isinstance(components.get(name), int) and components[name] >= 1 for name in required_components)
    ):
        failures.append("heca_component_call_counters.json:schema")

    conservation_rows = _read_jsonl(root / HECA_SIDECAR_FILES["contribution_conservation"])
    conservation_fields = {"action", "sum_contribution", "action_credit_sum", "abs_error"}
    if not conservation_rows or not all(
        conservation_fields.issubset(row)
        and isinstance(row["action"], int)
        and 0 <= row["action"] < 4
        and all(_is_finite_number(row[field]) for field in conservation_fields - {"action"})
        and abs(float(row["sum_contribution"]) - float(row["action_credit_sum"])) <= 1e-5
        and float(row["abs_error"]) <= 1e-5
        for row in conservation_rows
    ):
        failures.append("heca_contribution_conservation.jsonl:schema")

    schedule = _read_json(root / HECA_SIDECAR_FILES["schedule_state"])
    excess_risk = schedule.get("excess_risk") if schedule else None
    schedule_fields = {"optimizer_step", "progress", "credit_ramp", "foundation_grad_cap"}
    if not (
        schedule
        and all(_is_finite_number(schedule.get(field)) for field in schedule_fields)
        and isinstance(excess_risk, dict)
        and all(_is_finite_number(excess_risk.get(name)) for name in ("action", "reason"))
    ):
        failures.append("heca_schedule_state.json:schema")

    ablation = _read_json(root / HECA_SIDECAR_FILES["ablation_manifest"])
    independent = ablation.get("independent_runs") if ablation else None
    if not (
        ablation
        and set(ablation.get("cheap_same_forward", [])) == set(HECA_CHEAP_MODE_NAMES)
        and isinstance(independent, dict)
        and set(independent) == {f"B{index}" for index in range(6)}
        and all(
            isinstance(item, dict) and item.get("execution") == "independent_run"
            for item in independent.values()
        )
    ):
        failures.append("heca_ablation_manifest.json:schema")

    for letter in HECA_GATE_NAMES:
        gate = _read_json(root / f"HECA_GATE_{letter}.json")
        if not (
            gate
            and gate.get("gate") == letter
            and isinstance(gate.get("pass"), bool)
            and isinstance(gate.get("evidence"), dict)
            and gate["evidence"]
        ):
            failures.append(f"HECA_GATE_{letter}.json:schema")
    return failures


def write_heca_pilot_evidence_manifest(
    directory: str | Path, *, git_head: str
) -> dict[str, Any]:
    """Bind Gate A-G to the exact pilot evidence files used to derive them."""
    root = Path(directory)
    relative_paths = [
        Path(name)
        for name in (
            *HECA_SIDECAR_FILES.values(),
            *HECA_PILOT_INPUT_FILES,
            *(f"HECA_GATE_{letter}.json" for letter in HECA_GATE_NAMES),
            "loss_components.jsonl",
        )
    ]
    for epoch in sorted(path for path in root.glob("epoch_*") if path.is_dir()):
        relative_paths.extend(
            epoch.relative_to(root) / name
            for name in ("branch_metrics.json", "typed_evidence.json", "runtime.json")
        )
    epoch_branches = [
        path for path in relative_paths if path.name == "branch_metrics.json"
    ]
    missing = [str(path) for path in relative_paths if not (root / path).exists()]
    if missing or len(epoch_branches) != 4:
        raise ValueError(
            "HECA pilot evidence is incomplete: "
            + (", ".join(missing) if missing else "expected exactly four epochs")
        )
    payload = {
        "git_head": str(git_head),
        "files": {
            path.as_posix(): file_hash(root / path)
            for path in sorted(set(relative_paths))
        },
    }
    write_json(root / HECA_PILOT_EVIDENCE_MANIFEST, payload)
    return payload


def validate_heca_pilot_bundle(
    directory: str | Path, *, expected_git_head: str
) -> list[str]:
    """Validate sidecar, current HEAD binding, and every raw evidence hash."""
    root = Path(directory)
    failures = validate_heca_artifact_sidecar(root)
    manifest = _read_json(root / HECA_PILOT_EVIDENCE_MANIFEST)
    pilot = _read_json(root / "HECA_PILOT_PASS.json")
    if not manifest:
        failures.append(f"{HECA_PILOT_EVIDENCE_MANIFEST}:missing_or_invalid")
        return failures
    if manifest.get("git_head") != expected_git_head:
        failures.append(f"{HECA_PILOT_EVIDENCE_MANIFEST}:git_head")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        failures.append(f"{HECA_PILOT_EVIDENCE_MANIFEST}:files")
    else:
        epoch_branches = [
            name for name in files if name.endswith("/branch_metrics.json")
        ]
        if len(epoch_branches) != 4:
            failures.append(f"{HECA_PILOT_EVIDENCE_MANIFEST}:four_epoch_evidence")
        for name, expected_hash in files.items():
            path = root / str(name)
            if not path.exists():
                failures.append(f"{name}:missing")
            elif not isinstance(expected_hash, str) or file_hash(path) != expected_hash:
                failures.append(f"{name}:hash")
    if not pilot:
        failures.append("HECA_PILOT_PASS.json:missing_or_invalid")
    else:
        if pilot.get("pass") is not True or pilot.get("git_head") != expected_git_head:
            failures.append("HECA_PILOT_PASS.json:status_or_head")
        if pilot.get("evidence_manifest_sha256") != file_hash(
            root / HECA_PILOT_EVIDENCE_MANIFEST
        ):
            failures.append("HECA_PILOT_PASS.json:manifest_hash")
    return failures


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
    has_heca_sidecar = any(
        (root / name).exists()
        for name in (*HECA_SIDECAR_FILES.values(), *(f"HECA_GATE_{letter}.json" for letter in HECA_GATE_NAMES))
    )
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
        if has_heca_sidecar:
            failures.extend(validate_heca_artifact_sidecar(root))
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
    factor_rows = train_audit.get("per_factor", []) if isinstance(train_audit, dict) else []
    heca_rows = [
        row for row in factor_rows
        if isinstance(row, dict) and "observability_visually_unidentifiable" in row
    ]
    if heca_rows:
        def _valid_heca_factor_row(row: dict[str, Any]) -> bool:
            positive = row.get("state_positive_count")
            negative = row.get("state_negative_count")
            identifiable = row.get("state_identifiable")
            return (
                row.get("audit_split") == "train_audit"
                and isinstance(positive, int)
                and isinstance(negative, int)
                and positive >= 0
                and negative >= 0
                and isinstance(identifiable, bool)
                and identifiable == (positive >= 20 and negative >= 20)
            )

        typed_valid = typed_valid and len(heca_rows) == 21 and all(
            _valid_heca_factor_row(row) for row in heca_rows
        )
    # Formal TESA artifacts never fall back to the historical weak schema.
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
    if has_heca_sidecar:
        failures.extend(validate_heca_artifact_sidecar(root))
    return failures
