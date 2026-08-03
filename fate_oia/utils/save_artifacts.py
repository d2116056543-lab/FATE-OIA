from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


MANIFEST_HASH_KEYS = (
    "git_head",
    "config_hash",
    "source_tree_hash",
    "schema_hash",
    "split_hash",
    "checkpoint_hash",
    "logits_hash",
    "labels_hash",
    "file_order_hash",
)

_CHECKPOINT_REQUIRED = (
    "model",
    "optimizer",
    "scheduler",
    "optimizer_step",
    "rng_state",
    "action_rms_ema",
    "view_consistency_ema",
    "utility_cadence",
    "utility_cadence_phase",
    "tail_prototypes",
    "pu_lambda",
    "calibration",
    "split_manifest",
    "git_head",
    "config_hash",
    "source_tree_hash",
    "schema_hash",
    "split_hash",
    "file_order_hash",
)


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
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_safe(value),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def hash_value(value: Any) -> str:
    """Return a stable SHA-256 for JSON-compatible metadata."""
    return _sha256_bytes(_json_bytes(value))


def save_source_tree_hash(root: str | Path) -> str:
    """Hash every SAVE surface that can change the formal run semantics."""
    source_root = Path(root).resolve()
    candidates: list[Path] = []
    candidates.extend((source_root / "fate_oia").rglob("save_*.py"))
    candidates.extend((source_root / "configs").glob("*save*.yaml"))
    candidates.append(source_root / "configs" / "save_factor_schema.yaml")
    candidates.extend((source_root / "scripts").glob("FATE_OIA_save_oia_v1_*.ps1"))
    skill_root = source_root / ".codex" / "skills" / "save-oia-implementation-audit"
    if skill_root.exists():
        candidates.extend(skill_root.rglob("*.md"))
    digest = hashlib.sha256()
    for path in sorted({path.resolve() for path in candidates if path.is_file()}, key=str):
        digest.update(str(path.relative_to(source_root)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: str | Path, data: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return target


def write_json(path: str | Path, value: Any) -> Path:
    return _atomic_write(path, _json_bytes(value) + b"\n")


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> Path:
    data = b"".join(_json_bytes(row) + b"\n" for row in rows)
    return _atomic_write(path, data)


def append_jsonl(path: str | Path, value: Any) -> Path:
    target = Path(path)
    rows: list[Any] = []
    if target.exists():
        try:
            rows = [
                json.loads(line)
                for line in target.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSONL artifact: {target}") from exc
    rows.append(value)
    return write_jsonl(target, rows)


def _torch_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def save_tensor(path: str | Path, value: torch.Tensor) -> Path:
    if not isinstance(value, torch.Tensor):
        raise TypeError("artifact tensors must be torch.Tensor values")
    return _atomic_write(path, _torch_bytes(value.detach().cpu()))


save_meter_tensor = save_tensor


def _hash_named_tensors(values: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        if not isinstance(values[name], torch.Tensor):
            raise TypeError(f"tensor collection contains non-tensor value: {name}")
        name_bytes = str(name).encode("utf-8")
        payload = _torch_bytes(values[name].detach().cpu())
        digest.update(len(name_bytes).to_bytes(8, "big"))
        digest.update(name_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _hash_file_order(file_order: Iterable[str]) -> str:
    return hash_value({"file_order": [str(name) for name in file_order]})


def _provided_hash(
    hashes: Mapping[str, Any],
    *names: str,
    default: str | None = None,
) -> str:
    for name in names:
        value = hashes.get(name)
        if value is not None:
            return str(value)
    if default is not None:
        return str(default)
    return hash_value(None)


def _split_hash(split_manifest: Any, explicit: Any, file_order: list[str]) -> str:
    if explicit is not None:
        return str(explicit)
    if split_manifest is not None:
        return hash_value(split_manifest)
    return hash_value({"file_order": file_order})


def _resolve_metadata_hash(explicit: Any, value: Any) -> str:
    if explicit is not None:
        return str(explicit) if isinstance(explicit, str) else hash_value(explicit)
    if value is not None:
        return str(value) if isinstance(value, str) else hash_value(value)
    return hash_value(None)


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    required = {"python", "numpy", "torch", "cuda"}
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"checkpoint RNG state missing: {', '.join(missing)}")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    epoch: int = 0,
    micro_step: int = 0,
    optimizer_step: int,
    runtime_profile: Mapping[str, Any] | None = None,
    meta_state: Mapping[str, Any] | None = None,
    pu_state: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    config_hash: Any = None,
    source_hash: Any = None,
    schema_hash: Any = None,
    action_rms_ema: Any = None,
    view_consistency_ema: Any = None,
    utility_cadence: Any = None,
    tail_prototypes: Any = None,
    pu_lambda: Any = None,
    split_manifest: Any = None,
    config: Any = None,
    source: Any = None,
    schema: Any = None,
    git_head: str | None = None,
    source_tree_hash: Any = None,
    split_hash: Any = None,
    file_order_hash: Any = None,
) -> Path:
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if optimizer_step < 0:
        raise ValueError("optimizer_step must be non-negative")
    source_tree = _resolve_metadata_hash(source_tree_hash, source_hash)
    if source_tree_hash is None and source_hash is None and source is not None:
        source_tree = _resolve_metadata_hash(None, source)
    resolved_git_head = str(git_head or (source if isinstance(source, str) else source_tree))
    resolved_config = _resolve_metadata_hash(config_hash, config)
    resolved_schema = _resolve_metadata_hash(schema_hash, schema)
    resolved_split = _split_hash(split_manifest, split_hash, [])
    resolved_file_order = str(file_order_hash or hash_value([]))
    cadence = copy.deepcopy(utility_cadence if utility_cadence is not None else {})
    pu_value = copy.deepcopy(pu_lambda)
    if pu_value is None and isinstance(pu_state, Mapping):
        pu_value = copy.deepcopy(pu_state.get("lambda", {}))
    if pu_value is None:
        pu_value = {}
    payload = {
        "format": "fate_oia.save_checkpoint.v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "micro_step": int(micro_step),
        "optimizer_step": int(optimizer_step),
        "rng_state": capture_rng_state(),
        "action_rms_ema": copy.deepcopy(action_rms_ema if action_rms_ema is not None else {}),
        "view_consistency_ema": copy.deepcopy(
            view_consistency_ema if view_consistency_ema is not None else {}
        ),
        "utility_cadence": cadence,
        "utility_cadence_phase": copy.deepcopy(cadence),
        "tail_prototypes": copy.deepcopy(tail_prototypes if tail_prototypes is not None else {}),
        "pu_lambda": pu_value,
        "calibration": copy.deepcopy(calibration if calibration is not None else {}),
        "split_manifest": copy.deepcopy(split_manifest if split_manifest is not None else {}),
        "git_head": resolved_git_head,
        "config_hash": resolved_config,
        "source_tree_hash": source_tree,
        "schema_hash": resolved_schema,
        "split_hash": resolved_split,
        "file_order_hash": resolved_file_order,
        "runtime_profile": copy.deepcopy(runtime_profile if runtime_profile is not None else {}),
        "meta_state": copy.deepcopy(meta_state if meta_state is not None else {}),
        "pu_state": copy.deepcopy(pu_state if pu_state is not None else {}),
        "source_hash": source_tree,
    }
    return _atomic_write(path, _torch_bytes(payload))


def _validate_checkpoint_metadata(
    payload: Mapping[str, Any],
    *,
    expected_git_head: str | None,
    expected_config_hash: str | None,
    expected_source_tree_hash: str | None,
    expected_schema_hash: str | None,
    expected_split_hash: str | None,
    expected_file_order_hash: str | None,
) -> None:
    missing = [key for key in _CHECKPOINT_REQUIRED if key not in payload]
    if missing:
        raise ValueError(f"checkpoint missing required field: {missing[0]}")
    for key in (
        "git_head",
        "config_hash",
        "source_tree_hash",
        "schema_hash",
        "split_hash",
        "file_order_hash",
    ):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ValueError(f"checkpoint {key} is missing or invalid")
    expected = {
        "git_head": expected_git_head,
        "config_hash": expected_config_hash,
        "source_tree_hash": expected_source_tree_hash,
        "schema_hash": expected_schema_hash,
        "split_hash": expected_split_hash,
        "file_order_hash": expected_file_order_hash,
    }
    for key, value in expected.items():
        if value is not None and payload[key] != value:
            raise ValueError(f"Checkpoint {key} mismatch")
    if hash_value(payload["split_manifest"]) != payload["split_hash"]:
        if payload["split_manifest"] != {}:
            raise ValueError("checkpoint split_hash mismatch")


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    expected_git_head: str | None = None,
    expected_config_hash: str | None = None,
    expected_source_tree_hash: str | None = None,
    expected_source_hash: str | None = None,
    expected_schema_hash: str | None = None,
    expected_split_hash: str | None = None,
    expected_file_order_hash: str | None = None,
    expected_source: str | None = None,
) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ValueError(f"unable to read checkpoint: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload must be a mapping")
    _validate_checkpoint_metadata(
        payload,
        expected_git_head=expected_git_head or expected_source,
        expected_config_hash=expected_config_hash,
        expected_source_tree_hash=expected_source_tree_hash or expected_source_hash,
        expected_schema_hash=expected_schema_hash,
        expected_split_hash=expected_split_hash,
        expected_file_order_hash=expected_file_order_hash,
    )
    try:
        model.load_state_dict(payload["model"])
        if optimizer is not None:
            if payload["optimizer"] is None:
                raise ValueError("checkpoint optimizer state is absent")
            optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None:
            if payload["scheduler"] is None:
                raise ValueError("checkpoint scheduler state is absent")
            scheduler.load_state_dict(payload["scheduler"])
        restore_rng_state(payload["rng_state"])
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"checkpoint state restoration failed: {path}") from exc
    return dict(payload)


def save_epoch_artifacts(
    root: str | Path,
    epoch: int,
    *,
    metrics_raw: Mapping[str, Any],
    metrics_deploy: Mapping[str, Any],
    branch_metrics: Mapping[str, Any] | None = None,
    logits: Mapping[str, torch.Tensor],
    labels: Mapping[str, torch.Tensor],
    file_names: Iterable[str] | None = None,
    file_order: Iterable[str] | None = None,
    mechanism: Mapping[str, Any] | None = None,
    utility: Mapping[str, Any] | None = None,
    faithfulness: Mapping[str, Any] | None = None,
    gradient: Mapping[str, Any] | None = None,
    grad: Mapping[str, Any] | None = None,
    runtime: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    hashes: Mapping[str, Any] | None = None,
    git_head: str | None = None,
    config_hash: Any = None,
    source_tree_hash: Any = None,
    schema_hash: Any = None,
    split_manifest: Any = None,
    checkpoint: str | Path | Mapping[str, Any] | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_hash: str | None = None,
    split_hash: str | None = None,
) -> Path:
    directory = Path(root) / f"epoch_{int(epoch):03d}"
    directory.mkdir(parents=True, exist_ok=True)
    order = [str(name) for name in (file_order if file_order is not None else file_names or [])]
    if file_order is None and file_names is None:
        raise ValueError("artifact requires file_names or file_order")
    if not isinstance(logits, Mapping) or not logits:
        raise ValueError("artifact requires all branch logits")
    if not isinstance(labels, Mapping) or not labels:
        raise ValueError("artifact requires labels")
    write_json(directory / "metrics_raw.json", metrics_raw)
    write_json(directory / "metrics_deploy.json", metrics_deploy)
    if branch_metrics is not None:
        write_json(directory / "branch_metrics.json", branch_metrics)
    write_json(directory / "file_order.json", {"file_order": order})
    write_json(directory / "file_names_test.json", {"file_names": order})
    for name, value in logits.items():
        save_tensor(directory / f"logits_{name}.pt", value)
    for name, value in labels.items():
        save_tensor(directory / f"labels_{name}.pt", value)
    diagnostics = diagnostics or {}
    groups: dict[str, Any] = {
        "mechanism": mechanism,
        "utility": utility,
        "faithfulness": faithfulness,
        "gradient": gradient if gradient is not None else grad,
        "runtime": runtime,
    }
    for name, value in diagnostics.items():
        filename = str(name)
        if filename.endswith(".json") or filename.endswith(".jsonl"):
            filename = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        else:
            filename = f"{filename}.json"
        write_json(directory / filename, value)
    for name, value in groups.items():
        if value is None:
            raise ValueError(f"artifact missing required field: {name}")
        write_json(directory / f"{name}.json", value)

    provided = dict(hashes or {})
    resolved_git = str(git_head or _provided_hash(provided, "git_head", "source", default="unknown"))
    resolved_config = _provided_hash(provided, "config_hash", "config", default=config_hash)
    resolved_source = _provided_hash(
        provided, "source_tree_hash", "source", default=source_tree_hash
    )
    resolved_schema = _provided_hash(provided, "schema_hash", "schema", default=schema_hash)
    resolved_split = _provided_hash(
        provided,
        "split_hash",
        default=split_hash if split_hash is not None else _split_hash(split_manifest, None, order),
    )
    resolved_checkpoint = checkpoint_hash
    resolved_checkpoint_path = checkpoint_path
    if resolved_checkpoint_path is None and isinstance(checkpoint, (str, Path)):
        resolved_checkpoint_path = checkpoint
    if resolved_checkpoint is None and resolved_checkpoint_path is not None:
        resolved_checkpoint = file_hash(resolved_checkpoint_path)
    if resolved_checkpoint is None and isinstance(checkpoint, Mapping):
        resolved_checkpoint = hash_value(checkpoint)
    if resolved_checkpoint is None:
        resolved_checkpoint = _provided_hash(provided, "checkpoint_hash", "checkpoint")
    resolved_logits = _hash_named_tensors(logits)
    resolved_labels = _hash_named_tensors(labels)
    resolved_order = _hash_file_order(order)
    hash_payload = {
        "git_head": resolved_git,
        "config_hash": resolved_config,
        "source_tree_hash": resolved_source,
        "schema_hash": resolved_schema,
        "split_hash": resolved_split,
        "checkpoint_hash": resolved_checkpoint,
        "logits_hash": resolved_logits,
        "labels_hash": resolved_labels,
        "file_order_hash": resolved_order,
        "config": resolved_config,
        "source": resolved_source,
        "schema": resolved_schema,
        "file_order": resolved_order,
        "logits": resolved_logits,
        "labels": resolved_labels,
        "checkpoint": resolved_checkpoint,
    }
    write_json(
        directory / "manifest.json",
        {
            "format": "fate_oia.save_artifact.v1",
            "epoch": int(epoch),
            "hashes": hash_payload,
            "files": {
                "logits": sorted(f"logits_{name}.pt" for name in logits),
                "labels": sorted(f"labels_{name}.pt" for name in labels),
                "checkpoint": str(resolved_checkpoint_path) if resolved_checkpoint_path else None,
            },
        },
    )
    return directory


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON artifact: {path.name}") from exc


def validate_epoch_artifacts(directory: str | Path) -> bool:
    root = Path(directory)
    required_files = (
        "manifest.json",
        "metrics_raw.json",
        "metrics_deploy.json",
        "mechanism.json",
        "utility.json",
        "faithfulness.json",
        "gradient.json",
        "runtime.json",
        "file_order.json",
    )
    for name in required_files:
        if not (root / name).is_file():
            raise ValueError(f"artifact missing required field: {name}")
    manifest = _read_json(root / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise ValueError("artifact manifest schema is invalid")
    hashes = manifest.get("hashes")
    if not isinstance(hashes, Mapping):
        raise ValueError("artifact manifest hashes are missing")
    missing_hashes = [name for name in MANIFEST_HASH_KEYS if not hashes.get(name)]
    if missing_hashes:
        raise ValueError(f"artifact manifest missing hash: {missing_hashes[0]}")
    placeholder_hashes = [
        name for name in MANIFEST_HASH_KEYS
        if str(hashes.get(name, "")).lower() in {"pending", "unknown", "none"}
    ]
    if placeholder_hashes:
        raise ValueError(f"artifact manifest has placeholder hash: {placeholder_hashes[0]}")
    order_payload = _read_json(root / "file_order.json")
    order = order_payload.get("file_order") if isinstance(order_payload, Mapping) else None
    if not isinstance(order, list) or not all(isinstance(name, str) for name in order):
        raise ValueError("artifact file_order schema is invalid")
    files = manifest.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("artifact manifest files are missing")
    logit_files = files.get("logits")
    label_files = files.get("labels")
    if not isinstance(logit_files, list) or not logit_files:
        raise ValueError("artifact logits files are missing")
    if not isinstance(label_files, list) or not label_files:
        raise ValueError("artifact labels files are missing")
    logits: dict[str, torch.Tensor] = {}
    labels: dict[str, torch.Tensor] = {}
    for filename in logit_files:
        if not isinstance(filename, str) or not filename.startswith("logits_"):
            raise ValueError("artifact logits manifest is invalid")
        path = root / filename
        if not path.is_file():
            raise ValueError(f"artifact missing required field: {filename}")
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(f"artifact logits unreadable: {filename}") from exc
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"artifact logits schema is invalid: {filename}")
        logits[filename[len("logits_") : -len(".pt")]] = value
    for filename in label_files:
        if not isinstance(filename, str) or not filename.startswith("labels_"):
            raise ValueError("artifact labels manifest is invalid")
        path = root / filename
        if not path.is_file():
            raise ValueError(f"artifact missing required field: {filename}")
        try:
            value = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:
            raise ValueError(f"artifact labels unreadable: {filename}") from exc
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"artifact labels schema is invalid: {filename}")
        labels[filename[len("labels_") : -len(".pt")]] = value
    actual = {
        "file_order_hash": _hash_file_order(order),
        "logits_hash": _hash_named_tensors(logits),
        "labels_hash": _hash_named_tensors(labels),
    }
    for name, value in actual.items():
        if hashes.get(name) != value:
            raise ValueError(f"artifact {name} mismatch")
    return True


__all__ = [
    "MANIFEST_HASH_KEYS",
    "append_jsonl",
    "capture_rng_state",
    "file_hash",
    "hash_value",
    "load_checkpoint",
    "restore_rng_state",
    "save_checkpoint",
    "save_epoch_artifacts",
    "save_meter_tensor",
    "save_tensor",
    "validate_epoch_artifacts",
    "write_json",
    "write_jsonl",
]
