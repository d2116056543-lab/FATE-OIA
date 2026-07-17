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
ICDOR_ROOT_JSONL_FILES = ("metrics_summary.jsonl", "adaptive_schedule.jsonl")
ICDOR_EPOCH_JSON_FILES = (
    "metrics_summary.json",
    "branch_metrics.json",
    "per_label_metrics.json",
    "factor_certificate_snapshot.json",
    "visual_credibility.json",
    "target_transfer_summary.json",
    "semantic_compatibility.json",
    "target_utility.json",
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
    "credibility_stats.jsonl",
    "fine_transport_stats.jsonl",
    "route_ownership.jsonl",
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
    _append_jsonl(output / "adaptive_schedule.jsonl", {
        "event": "initialized", "state_after": "FOUNDATION", "state_epochs_after": 0,
        "source_splits": ["train_core", "audit_visual", "audit_target", "train_calib"],
    })
    return output


def write_icdor_adaptive_schedule_transition(
    output_dir: str | Path,
    transition: dict[str, Any],
) -> Path:
    """Append a source-sealed adaptive state transition for later audit."""
    required = {
        "epoch", "state_before", "state_after", "state_epochs_before", "state_epochs_after",
        "ready", "failed_closed", "readiness", "certificate_sha256", "edge_admission_sha256",
    }
    if not required <= set(transition):
        raise ValueError("IC-DOR adaptive schedule transition is incomplete")
    readiness = transition["readiness"]
    if not isinstance(readiness, dict) or set(readiness) != {"train_core", "train_audit", "train_calib"}:
        raise ValueError("IC-DOR adaptive schedule readiness must contain only train_core/train_audit/train_calib")
    for split, metrics in readiness.items():
        if not isinstance(metrics, dict) or metrics.get("source_split") != split:
            actual = metrics.get("source_split") if isinstance(metrics, dict) else type(metrics).__name__
            raise ValueError(
                f"IC-DOR adaptive schedule readiness for {split} has invalid provenance: {actual}"
            )
        if any(value == "test" for value in metrics.values()):
            raise ValueError("IC-DOR adaptive schedule artifacts must not contain test provenance")
    path = Path(output_dir) / "adaptive_schedule.jsonl"
    _append_jsonl(path, transition)
    return path


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


def _matched_control_provenance_valid(arms: Any) -> bool:
    if not isinstance(arms, list) or len(arms) < 4:
        return False
    identity = [arm for arm in arms if arm.get("control_type") == "same_type_identity"]
    spatial = [arm for arm in arms if arm.get("control_type") == "spatial_roll"]
    if len(identity) != 1 or len(spatial) < 3:
        return False
    if any(int(arm.get("available_sample_count", 0)) <= 0 for arm in arms):
        return False
    identity_arm = identity[0]
    identity_names = identity_arm.get("identity_source_factor_names")
    identity_types = identity_arm.get("identity_source_factor_types")
    identity_regions = identity_arm.get("identity_source_regions")
    selected_type = identity_arm.get("factor_type")
    selected_region = identity_arm.get("region")
    if (
        not isinstance(identity_names, list) or not identity_names
        or not isinstance(identity_types, list) or len(identity_types) != len(identity_names)
        or not isinstance(identity_regions, list) or len(identity_regions) != len(identity_names)
        or not selected_type or not selected_region
        or any(value != selected_type for value in identity_types)
        or any(value != selected_region for value in identity_regions)
    ):
        return False
    for arm in spatial:
        offsets = arm.get("spatial_offsets")
        if not isinstance(offsets, list) or not offsets:
            return False
        if any(
            not isinstance(offset, (list, tuple))
            or len(offset) != 2
            or not all(isinstance(value, int) for value in offset)
            or tuple(offset) == (0, 0)
            for offset in offsets
        ):
            return False
        if arm.get("factor_type") != selected_type or arm.get("region") != selected_region:
            return False
    return all(
        arm.get("max_mass_error") is not None
        and float(arm["max_mass_error"]) <= 0.05
        and arm.get("max_overlap") is not None
        and float(arm["max_overlap"]) == 0.0
        for arm in arms
    )


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
    v4_run = manifest.get("credo_version") == "v4_credo"
    split_manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    if v4_run:
        all_names = split_manifest.get("file_names")
        visual_ids = split_manifest.get("audit_visual_indices")
        target_ids = split_manifest.get("audit_target_indices")
        core_ids = split_manifest.get("train_core_indices")
        calib_ids = split_manifest.get("train_calib_indices")
        if not all(isinstance(value, list) for value in (all_names, visual_ids, target_ids, core_ids, calib_ids)):
            errors.append("split_manifest must expose list-valued core/audit/calibration partitions")
        else:
            total = len(all_names)
            expected_visual = max(1, round(total * 0.05))
            expected_target = max(1, round(total * 0.05))
            if len(visual_ids) != expected_visual or len(target_ids) != expected_target:
                errors.append("split_manifest audit_visual/audit_target must each be exactly 5 percent")
            visual_set, target_set = set(visual_ids), set(target_ids)
            core_set, calib_set = set(core_ids), set(calib_ids)
            if visual_set & target_set or visual_set & core_set or target_set & core_set:
                errors.append("split_manifest audit populations must be mutually disjoint")
            if visual_set | target_set | core_set | calib_set != set(range(total)):
                errors.append("split_manifest partitions must cover every train sample exactly once")
    certificate = json.loads((output / "factor_certificate.json").read_text(encoding="utf-8"))
    edge_admission = json.loads((output / "edge_admission.json").read_text(encoding="utf-8"))
    pilot_run = manifest.get("pilot") is True
    expected_certificate_source = "audit_visual" if v4_run else "train_audit"
    expected_edge_source = "audit_target" if v4_run else "train_audit"
    certificate_pending = certificate.get("status") == "pending"
    edge_pending = edge_admission.get("status") == "pending"
    if (
        (not (v4_run and pilot_run and certificate_pending) and certificate.get("source_split") != expected_certificate_source)
        or (not (v4_run and pilot_run and edge_pending) and edge_admission.get("source_split") != expected_edge_source)
    ):
        errors.append("certificate/edge admission provenance does not match the run version")
    for name, entry in (edge_admission.get("entries") or {}).items():
        if not isinstance(entry, dict) or entry.get("accepted") is not True:
            continue
        metrics = entry.get("metrics")
        if (
            not isinstance(metrics, dict)
            or float(metrics.get("tes_identity_lcb95", 0.0)) <= 0.0
            or float(metrics.get("tes_spatial_lcb95", 0.0)) <= 0.0
        ):
            errors.append(f"accepted edge {name} lacks positive identity/spatial intervention LCBs")
    schedule_rows = _read_jsonl(output / "adaptive_schedule.jsonl")
    schedule_by_epoch = {
        int(row["epoch"]): row for row in schedule_rows
        if isinstance(row.get("epoch"), int)
    }
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
        reason_rows = _read_jsonl(epoch_dir / "reason_dual_observation_stats.jsonl")
        hidden_rows = [row for row in reason_rows if row.get("audit") == "hidden_recovery"]
        hidden_grid = {
            (row.get("mode"), float(row.get("hide_fraction", -1.0)))
            for row in hidden_rows
            if row.get("source_split") == ("audit_target" if v4_run else "train_audit") and row.get("evaluation_only") is True
        }
        expected_hidden_grid = {
            (mode, fraction)
            for mode in ("mcar", "mar", "mnar")
            for fraction in (0.10, 0.30, 0.50)
        }
        if hidden_grid != expected_hidden_grid or len(hidden_rows) != len(expected_hidden_grid):
            errors.append(f"epoch_{epoch:03d} hidden recovery lacks the leakage-free 10/30/50 audit grid")
        transfer = json.loads((epoch_dir / "target_transfer_summary.json").read_text(encoding="utf-8"))
        transfer_rows = _read_jsonl(epoch_dir / "target_transfer_stats.jsonl")
        transfer_fields = {"factor_id", "target_id", "tet", "tes", "cca", "ap_delta"}
        state_before = schedule_by_epoch.get(epoch, {}).get("state_before")
        if state_before == "FOUNDATION":
            if (
                transfer.get("available") is not False
                or transfer.get("source_split") != ("audit_target" if v4_run else "train_audit")
                or transfer.get("reason") != "interventions_disabled_in_foundation"
                or len(transfer_rows) != 1
                or transfer_rows[0].get("available") is not False
            ):
                errors.append(f"epoch_{epoch:03d} foundation target transfer did not abstain honestly")
        else:
            if (
                transfer.get("available") is not True
                or transfer.get("source_split") != ("audit_target" if v4_run else "train_audit")
                or transfer.get("schema_version") != "mosaic_target_transfer.v2"
                or int(transfer.get("pair_count", transfer.get("target_count", 0))) <= 0
            ):
                errors.append(f"epoch_{epoch:03d} target transfer is unavailable or not a real train_audit measurement")
            if not transfer_rows or any(
                row.get("available") is not True
                or row.get("source_split") != ("audit_target" if v4_run else "train_audit")
                or not transfer_fields <= set(row)
                or row.get("tes_identity") is None
                or row.get("tes_spatial") is None
                or not _matched_control_provenance_valid(row.get("matched_control_arms"))
                for row in transfer_rows
            ):
                errors.append(f"epoch_{epoch:03d} target transfer rows are incomplete")
        visual_credibility = json.loads((epoch_dir / "visual_credibility.json").read_text(encoding="utf-8"))
        if v4_run and (
            visual_credibility.get("source_split") != "audit_visual"
            or not isinstance(visual_credibility.get("credibility"), list)
        ):
            errors.append(f"epoch_{epoch:03d} visual credibility lacks audit_visual provenance")
        semantic = json.loads((epoch_dir / "semantic_compatibility.json").read_text(encoding="utf-8"))
        utility = json.loads((epoch_dir / "target_utility.json").read_text(encoding="utf-8"))
        if v4_run and state_before != "FOUNDATION":
            if (
                semantic.get("source_split") != "audit_target"
                or utility.get("source_split") != "audit_target"
                or semantic.get("available") is not True
                or utility.get("available") is not True
                or not isinstance(semantic.get("semantic_compatibility"), list)
                or not isinstance(utility.get("action_target_utility"), list)
            ):
                errors.append(f"epoch_{epoch:03d} target utility artifacts are incomplete or have invalid provenance")
        visual = json.loads((epoch_dir / "visual_audit_manifest.json").read_text(encoding="utf-8"))
        samples = visual.get("samples")
        if (
            visual.get("source_split") != ("audit_visual" if v4_run else "train_audit")
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

        # v4 diagnostics are semantic contracts, not merely non-empty files.
        credibility_rows = _read_jsonl(epoch_dir / "credibility_stats.jsonl")
        if v4_run and (not credibility_rows or any(
            row.get("split") != "test"
            or not isinstance(row.get("factor_id"), int)
            or any(
                not isinstance(row.get(field), (int, float))
                or not torch.isfinite(torch.tensor(float(row[field])))
                or not 0.0 <= float(row[field]) <= 1.0
                for field in ("cV_mean", "cV_p50", "cV_p95", "cV_ema_mean", "cV_nonzero_rate")
            )
            for row in credibility_rows
        )):
            errors.append(f"epoch_{epoch:03d} credibility_stats lacks finite bounded cV rows")
        fine_rows = _read_jsonl(epoch_dir / "fine_transport_stats.jsonl")
        if v4_run and (not fine_rows or any(
            row.get("split") != "test"
            or row.get("typed_coordinates_present") is not True
            or not all(
                isinstance(row.get(field), (int, float))
                and torch.isfinite(torch.tensor(float(row[field])))
                for field in ("fine_mask_delta_mean", "fine_mask_delta_max", "anchor_separation_mean")
            )
            for row in fine_rows
        ) or not any(float(row.get("fine_mask_delta_mean", 0.0)) > 1e-8 for row in fine_rows)):
            errors.append(f"epoch_{epoch:03d} fine_transport_stats does not prove typed fine evidence differs from coarse")
        route_rows = _read_jsonl(epoch_dir / "route_ownership.jsonl")
        if v4_run and (not route_rows or not any(row.get("summary") == "per_action_route_effect" for row in route_rows)):
            errors.append(f"epoch_{epoch:03d} route_ownership lacks per-action ownership diagnostics")
        if v4_run:
            for row in route_rows:
                if row.get("route_mode") == "shadow" and row.get("action_final_visual_equal") is not True:
                    errors.append(f"epoch_{epoch:03d} shadow route changed final action before admission")
                    break
    result["semantic_errors"] = errors
    result["pass"] = result["pass"] and not errors
    return result
