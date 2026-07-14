from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


ICDOR_ROOT_JSON_FILES = (
    "run_manifest.json",
    "source_manifest.json",
    "split_manifest.json",
    "runtime_selection.json",
    "factor_certificate.json",
    "edge_admission.json",
)
ICDOR_ROOT_JSONL_FILES = ("metrics_summary.jsonl",)
ICDOR_EPOCH_JSON_FILES = (
    "metrics_summary.json",
    "branch_metrics.json",
    "per_label_metrics.json",
    "factor_certificate_snapshot.json",
    "target_transfer_summary.json",
    "visual_audit_manifest.json",
)
ICDOR_EPOCH_JSONL_FILES = (
    "loss_components.jsonl",
    "factor_stats.jsonl",
    "prototype_stats.jsonl",
    "action_route_stats.jsonl",
    "reason_dual_observation_stats.jsonl",
    "target_transfer_stats.jsonl",
    "pareto_stats.jsonl",
    "gradient_ownership.jsonl",
    "calibration_stats.jsonl",
    "runtime_stats.jsonl",
    "failure_cases.jsonl",
)
ICDOR_LOGIT_FILES = (
    "action_visual_logits.pt",
    "action_shadow_logits.pt",
    "action_final_logits.pt",
    "action_deploy_logits.pt",
    "reason_visual_observed_logits.pt",
    "reason_latent_logits.pt",
    "reason_observation_model_prob.pt",
    "reason_observed_logits.pt",
    "reason_deploy_logits.pt",
    "action_labels.pt",
    "reason_labels.pt",
)


def _safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"IC-DOR artifact cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    if not payload:
        raise ValueError(f"IC-DOR JSON artifact {path.name} must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    if not row:
        raise ValueError(f"IC-DOR JSONL artifact {path.name} must not contain an empty row")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_safe(row), sort_keys=True) + "\n")


def initialize_icdor_run_artifacts(
    output_dir: str | Path,
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    source_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
    runtime_selection: dict[str, Any],
    factor_certificate: dict[str, Any],
    edge_admission: dict[str, Any],
) -> Path:
    """Write immutable run provenance before the first optimizer step."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "run_manifest.json", manifest)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(_safe(config), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_json(output / "source_manifest.json", source_manifest)
    _write_json(output / "split_manifest.json", split_manifest)
    _write_json(output / "runtime_selection.json", runtime_selection)
    _write_json(output / "factor_certificate.json", factor_certificate)
    _write_json(output / "edge_admission.json", edge_admission)
    return output


def write_icdor_epoch_artifacts(
    output_dir: str | Path,
    *,
    epoch: int,
    json_payloads: dict[str, dict[str, Any]],
    jsonl_payloads: dict[str, list[dict[str, Any]]],
    logits: dict[str, torch.Tensor],
    file_names: list[str],
) -> Path:
    """Fail closed unless every IC-DOR interpretation surface is populated."""
    if type(epoch) is not int or epoch < 0:
        raise ValueError("IC-DOR epoch must be a non-negative integer")
    if set(json_payloads) != set(ICDOR_EPOCH_JSON_FILES):
        raise ValueError("IC-DOR epoch JSON schema is incomplete or contains an unknown artifact")
    if set(jsonl_payloads) != set(ICDOR_EPOCH_JSONL_FILES):
        raise ValueError("IC-DOR epoch JSONL schema is incomplete or contains an unknown artifact")
    if set(logits) != set(ICDOR_LOGIT_FILES):
        raise ValueError("IC-DOR epoch logits schema is incomplete or contains an unknown artifact")
    sample_count = len(file_names)
    if sample_count <= 0 or any(not isinstance(name, str) or not name for name in file_names):
        raise ValueError("IC-DOR epoch file_names must be non-empty strings")
    for name, tensor in logits.items():
        if not isinstance(tensor, torch.Tensor) or tensor.ndim < 1 or tensor.shape[0] != sample_count:
            raise ValueError(f"IC-DOR {name} must align to file_names")
        if tensor.numel() == 0 or not torch.isfinite(tensor).all():
            raise ValueError(f"IC-DOR {name} is empty or non-finite")
    output = Path(output_dir)
    epoch_dir = output / f"epoch_{epoch:03d}"
    logits_dir = epoch_dir / "logits"
    logits_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in json_payloads.items():
        _write_json(epoch_dir / name, payload)
    for name, rows in jsonl_payloads.items():
        if not rows:
            raise ValueError(f"IC-DOR {name} requires at least one real diagnostic row")
        for row in rows:
            _append_jsonl(epoch_dir / name, row)
    for name, tensor in logits.items():
        torch.save(tensor.detach().cpu(), logits_dir / name)
    _write_json(logits_dir / "file_names.json", file_names)
    _append_jsonl(output / "metrics_summary.jsonl", json_payloads["metrics_summary.json"])
    return epoch_dir


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_icdor_artifact_schema(
    output_dir: str | Path,
    *,
    epochs: list[int],
    strict_semantics: bool = False,
    require_checkpoints: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    missing: list[str] = []
    invalid: list[str] = []
    for name in (*ICDOR_ROOT_JSON_FILES, "resolved_config.yaml", *ICDOR_ROOT_JSONL_FILES):
        path = output / name
        if not path.exists():
            missing.append(str(path))
        elif not path.stat().st_size:
            invalid.append(str(path))
    if require_checkpoints:
        for name in ("checkpoint_latest.pth", "checkpoint_best_test_joint.pth"):
            path = output / name
            if not path.exists() or not path.stat().st_size:
                missing.append(str(path))
    for epoch in epochs:
        epoch_dir = output / f"epoch_{epoch:03d}"
        for name in (*ICDOR_EPOCH_JSON_FILES, *ICDOR_EPOCH_JSONL_FILES):
            path = epoch_dir / name
            if not path.exists():
                missing.append(str(path))
            elif not path.stat().st_size:
                invalid.append(str(path))
        for name in (*ICDOR_LOGIT_FILES, "file_names.json"):
            path = epoch_dir / "logits" / name
            if not path.exists():
                missing.append(str(path))
            elif not path.stat().st_size:
                invalid.append(str(path))
    result: dict[str, Any] = {"pass": not missing and not invalid, "missing": missing, "invalid": invalid}
    if not strict_semantics or missing or invalid:
        return result
    errors: list[str] = []
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    for key, expected in (("direct_image", True), ("feature_cache", False), ("token_compression", "none"), ("best_selection_split", "test")):
        if manifest.get(key) != expected:
            errors.append(f"run_manifest.{key} must equal {expected!r}")
    if not manifest.get("git_head") or not manifest.get("pretrained_sha256"):
        errors.append("run_manifest must retain git_head and pretrained_sha256")
    certificate = json.loads((output / "factor_certificate.json").read_text(encoding="utf-8"))
    edge_admission = json.loads((output / "edge_admission.json").read_text(encoding="utf-8"))
    if certificate.get("source_split") != "train_audit" or edge_admission.get("source_split") != "train_audit":
        errors.append("certificate and edge admission must be train_audit-only")
    for epoch in epochs:
        epoch_dir = output / f"epoch_{epoch:03d}"
        metrics = json.loads((epoch_dir / "metrics_summary.json").read_text(encoding="utf-8"))
        if not {"raw", "deploy_fixed", "test_oracle_diagnostic"} <= set(metrics):
            errors.append(f"epoch_{epoch:03d} metrics lacks raw/deploy/oracle branches")
        if int(metrics.get("sample_count", 0)) <= 0:
            errors.append(f"epoch_{epoch:03d} metrics has no evaluated samples")
        branch = json.loads((epoch_dir / "branch_metrics.json").read_text(encoding="utf-8"))
        if branch.get("available") is not True:
            errors.append(f"epoch_{epoch:03d} branch metrics are unavailable")
        calibration_rows = _read_jsonl(epoch_dir / "calibration_stats.jsonl")
        if any(row.get("source_split") != "train_calib" for row in calibration_rows):
            errors.append(f"epoch_{epoch:03d} calibration uses a non-train_calib source")
        transfer = json.loads((epoch_dir / "target_transfer_summary.json").read_text(encoding="utf-8"))
        if (
            transfer.get("available") is not True
            or transfer.get("source_split") != "train_audit"
            or transfer.get("schema_version") != "mosaic_target_transfer.v1"
            or int(transfer.get("pair_count", transfer.get("target_count", 0))) <= 0
        ):
            errors.append(f"epoch_{epoch:03d} target transfer is unavailable or not a real train_audit measurement")
        transfer_rows = _read_jsonl(epoch_dir / "target_transfer_stats.jsonl")
        transfer_fields = {"factor_id", "target_id", "tet", "tes", "cca", "ap_delta"}
        if not transfer_rows or any(
            row.get("available") is not True
            or row.get("source_split") != "train_audit"
            or not transfer_fields <= set(row)
            for row in transfer_rows
        ):
            errors.append(f"epoch_{epoch:03d} target transfer rows are incomplete")
        visual = json.loads((epoch_dir / "visual_audit_manifest.json").read_text(encoding="utf-8"))
        samples = visual.get("samples")
        if (
            visual.get("source_split") != "train_audit"
            or visual.get("matched_random_control") != "same_factor_equal_mass_spatial_roll"
            or int(visual.get("sample_count", 0)) <= 0
            or not isinstance(samples, list)
            or len(samples) != int(visual.get("sample_count", 0))
        ):
            errors.append(f"epoch_{epoch:03d} visual matched-random audit is invalid")
        elif isinstance(samples, list):
            for sample in samples:
                original_files = sample.get("factor_mask_files")
                random_files = sample.get("matched_random_factor_mask_files")
                if (
                    not isinstance(original_files, list)
                    or not isinstance(random_files, list)
                    or len(original_files) == 0
                    or len(original_files) != len(random_files)
                ):
                    errors.append(f"epoch_{epoch:03d} visual mask file lists are invalid")
                    break
                for original_name, random_name in zip(original_files, random_files):
                    original_path, random_path = epoch_dir / original_name, epoch_dir / random_name
                    if not original_path.is_file() or not random_path.is_file():
                        errors.append(f"epoch_{epoch:03d} visual mask entity is missing")
                        break
                    original_mask = torch.load(original_path, map_location="cpu", weights_only=True)
                    random_mask = torch.load(random_path, map_location="cpu", weights_only=True)
                    expected = torch.roll(
                        original_mask,
                        shifts=(original_mask.shape[-2] // 3, original_mask.shape[-1] // 3),
                        dims=(-2, -1),
                    )
                    if not torch.equal(random_mask, expected) or not torch.isclose(original_mask.sum(), random_mask.sum()):
                        errors.append(f"epoch_{epoch:03d} visual matched-random mask is not the same-factor equal-mass roll")
                        break
        gradient_rows = _read_jsonl(epoch_dir / "gradient_ownership.jsonl")
        required_gradient_fields = {"epoch", "step", "loss", "owner_group", "grad_norm", "finite"}
        firewall_pairs = {
            ("loss_action_total", "reason_adapter"),
            ("loss_reason_total", "action_adapter"),
        }
        observed_pairs = {(row.get("loss"), row.get("owner_group")) for row in gradient_rows}
        if (
            not gradient_rows
            or any(not required_gradient_fields <= set(row) or row.get("finite") is not True for row in gradient_rows)
            or not firewall_pairs <= observed_pairs
            or any(float(row["grad_norm"]) != 0.0 for row in gradient_rows if (row.get("loss"), row.get("owner_group")) in firewall_pairs)
        ):
            errors.append(f"epoch_{epoch:03d} gradient ownership audit is incomplete")
        names = json.loads((epoch_dir / "logits" / "file_names.json").read_text(encoding="utf-8"))
        if not isinstance(names, list) or not names:
            errors.append(f"epoch_{epoch:03d} has no test file names")
        for name in ICDOR_LOGIT_FILES:
            tensor = torch.load(epoch_dir / "logits" / name, map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor) or not tensor.numel() or not torch.isfinite(tensor).all():
                errors.append(f"epoch_{epoch:03d}/{name} is empty or non-finite")
            elif tensor.shape[0] != len(names):
                errors.append(f"epoch_{epoch:03d}/{name} does not align with file_names")
    result["semantic_errors"] = errors
    result["pass"] = result["pass"] and not errors
    return result
