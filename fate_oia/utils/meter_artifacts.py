from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


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


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(value), indent=2, sort_keys=True, default=str), encoding="utf-8")


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True, default=str) + "\n")


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
    torch.save(payload, target)


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
        "per_action.json", "per_reason.json", "factor_stats.json",
        "evidence_maps_stats.json", "selector_stats.json", "reason_view_stats.json",
        "meta_stats.json", "pu_stats.json", "counterfactual.json", "calibration.json",
        "file_names_test.json",
        "logits_action_final_raw_test.pt", "logits_reason_final_raw_test.pt",
        "logits_action_visual_test.pt", "logits_action_semantic_test.pt",
        "logits_action_peer_test.pt", "logits_reason_calalign_test.pt",
        "logits_reason_global_test.pt", "logits_reason_local_test.pt",
        "logits_reason_mix_test.pt", "labels_action_test.pt", "labels_reason_test.pt",
    ]
    return [name for name in required if not (root / name).exists()]
