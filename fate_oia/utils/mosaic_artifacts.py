from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import yaml


ROOT_JSON_FILES = (
    "run_manifest.json",
    "git_state.json",
    "runtime_profile.json",
    "split_stats.json",
    "best_checkpoints.json",
)
ROOT_JSONL_FILES = ("metrics.jsonl", "supervisor_decisions.jsonl")
EPOCH_JSON_FILES = (
    "metrics_summary.json",
    "per_label_metrics.json",
    "action_branch_metrics.json",
    "reason_branch_metrics.json",
)
EPOCH_JSONL_FILES = (
    "loss_components.jsonl",
    "observable_factor_stats.jsonl",
    "factor_stats_by_factor.jsonl",
    "factor_grounding_stats.jsonl",
    "factor_mode_audit.jsonl",
    "prototype_usage_stats.jsonl",
    "decision_state_stats.jsonl",
    "selective_observation_stats.jsonl",
    "posterior_recovery_stats.jsonl",
    "action_rank_stats.jsonl",
    "reason_rank_stats.jsonl",
    "action_anchor_stats.jsonl",
    "threshold_stats.jsonl",
    "failure_cases.jsonl",
)
LOGIT_FILES = (
    "action_visual.pt",
    "action_state.pt",
    "action_raw.pt",
    "action_deploy.pt",
    "reason_latent.pt",
    "reason_deploy.pt",
    "labels_action.pt",
    "labels_reason.pt",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported artifact value type: {type(value)!r}")


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    if not payload:
        raise ValueError("JSONL artifact rows must not be empty placeholders")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")


def initialize_run_artifacts(
    output_dir: str | Path,
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    git_state: dict[str, Any],
    runtime_profile: dict[str, Any],
    split_stats: dict[str, Any],
) -> Path:
    output_dir = Path(output_dir)
    if not manifest or not config or not git_state or not runtime_profile or not split_stats:
        raise ValueError("run root artifacts require non-empty verified payloads")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "run_manifest.json", manifest)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(_json_safe(config), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    write_json(output_dir / "git_state.json", git_state)
    write_json(output_dir / "runtime_profile.json", runtime_profile)
    write_json(output_dir / "split_stats.json", split_stats)
    write_json(output_dir / "best_checkpoints.json", {"initialized": True, "records": {}})
    append_jsonl(output_dir / "supervisor_decisions.jsonl", {"event": "run_initialized"})
    return output_dir


def write_epoch_artifacts(
    output_dir: str | Path,
    *,
    epoch: int,
    json_payloads: dict[str, dict[str, Any]],
    jsonl_payloads: dict[str, list[dict[str, Any]]],
    logits: dict[str, torch.Tensor],
    sample_ids: list[str],
) -> Path:
    if type(epoch) is not int or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if set(json_payloads) != set(EPOCH_JSON_FILES):
        raise ValueError("epoch JSON artifact set does not match the MOSAIC schema")
    if set(jsonl_payloads) != set(EPOCH_JSONL_FILES):
        raise ValueError("epoch JSONL artifact set does not match the MOSAIC schema")
    if set(logits) != set(LOGIT_FILES):
        raise ValueError("epoch logit artifact set does not match the MOSAIC schema")
    batch_sizes = {int(tensor.shape[0]) for tensor in logits.values() if tensor.ndim > 0}
    if len(batch_sizes) != 1 or next(iter(batch_sizes)) != len(sample_ids):
        raise ValueError("epoch logits, labels, and sample IDs must have matching first dimensions")
    epoch_dir = Path(output_dir) / f"epoch_{epoch:03d}"
    logit_dir = epoch_dir / "logits"
    logit_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in json_payloads.items():
        if not payload:
            raise ValueError(f"epoch JSON artifact {name} must not be empty")
        write_json(epoch_dir / name, payload)
    for name, rows in jsonl_payloads.items():
        if not rows:
            raise ValueError(f"epoch JSONL artifact {name} must contain real rows")
        for row in rows:
            append_jsonl(epoch_dir / name, row)
    for name, tensor in logits.items():
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            raise ValueError(f"epoch tensor artifact {name} must be non-empty")
        torch.save(tensor.detach().cpu(), logit_dir / name)
    write_json(logit_dir / "sample_ids.json", sample_ids)
    append_jsonl(Path(output_dir) / "metrics.jsonl", json_payloads["metrics_summary.json"])
    return epoch_dir


def validate_artifact_schema(
    output_dir: str | Path,
    *,
    epochs: list[int],
    strict_semantics: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    missing: list[str] = []
    invalid: list[str] = []
    for name in (*ROOT_JSON_FILES, "resolved_config.yaml", *ROOT_JSONL_FILES):
        path = output_dir / name
        if not path.exists():
            missing.append(str(path))
        elif path.stat().st_size == 0:
            invalid.append(str(path))
    for epoch in epochs:
        epoch_dir = output_dir / f"epoch_{epoch:03d}"
        for name in (*EPOCH_JSON_FILES, *EPOCH_JSONL_FILES):
            path = epoch_dir / name
            if not path.exists():
                missing.append(str(path))
            elif path.stat().st_size == 0:
                invalid.append(str(path))
        for name in (*LOGIT_FILES, "sample_ids.json"):
            path = epoch_dir / "logits" / name
            if not path.exists():
                missing.append(str(path))
            elif path.stat().st_size == 0:
                invalid.append(str(path))
    result: dict[str, Any] = {"pass": not missing and not invalid, "missing": missing, "invalid": invalid}
    if not strict_semantics:
        return result

    semantic_errors: list[str] = []
    if not missing and not invalid:
        manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
        for key, expected in (
            ("direct_image", True),
            ("feature_cache", False),
            ("token_compression", "none"),
            ("best_selection_split", "test"),
        ):
            if manifest.get(key) != expected:
                semantic_errors.append(f"run_manifest.{key} must equal {expected!r}")
        if not manifest.get("git_head") or not manifest.get("pretrained_sha256"):
            semantic_errors.append("run_manifest requires git_head and pretrained_sha256")

        for epoch in epochs:
            epoch_dir = output_dir / f"epoch_{epoch:03d}"
            metrics = json.loads((epoch_dir / "metrics_summary.json").read_text(encoding="utf-8"))
            if not {"raw", "deploy_fixed", "test_oracle_diagnostic"} <= set(metrics):
                semantic_errors.append(f"epoch_{epoch:03d}/metrics_summary.json lacks metric branches")
            if int(metrics.get("sample_count", 0)) <= 0:
                semantic_errors.append(f"epoch_{epoch:03d}/metrics_summary.json has no samples")

            for name in EPOCH_JSONL_FILES:
                path = epoch_dir / name
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if not rows or any(not isinstance(row, dict) or not row for row in rows):
                    semantic_errors.append(f"epoch_{epoch:03d}/{name} has invalid rows")
                    continue
                unavailable = [row for row in rows if row.get("available") is False]
                if len(unavailable) == len(rows) and any(not row.get("reason") for row in unavailable):
                    semantic_errors.append(f"epoch_{epoch:03d}/{name} is placeholder-only without reason")
                if name == "threshold_stats.jsonl" and any(row.get("source") != "train_calib" for row in rows):
                    semantic_errors.append(f"epoch_{epoch:03d}/{name} has non-train_calib threshold source")

            sample_ids = json.loads(
                (epoch_dir / "logits" / "sample_ids.json").read_text(encoding="utf-8")
            )
            if not isinstance(sample_ids, list) or not sample_ids or any(not str(value) for value in sample_ids):
                semantic_errors.append(f"epoch_{epoch:03d}/logits/sample_ids.json is empty or invalid")
            for name in LOGIT_FILES:
                tensor = torch.load(epoch_dir / "logits" / name, map_location="cpu", weights_only=True)
                if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0 or not torch.isfinite(tensor).all():
                    semantic_errors.append(f"epoch_{epoch:03d}/logits/{name} is empty or non-finite")
                if name not in {"labels_action.pt", "labels_reason.pt"} and not torch.count_nonzero(tensor):
                    semantic_errors.append(f"epoch_{epoch:03d}/logits/{name} is an all-zero placeholder")

    result["semantic_errors"] = semantic_errors
    result["pass"] = result["pass"] and not semantic_errors
    return result
