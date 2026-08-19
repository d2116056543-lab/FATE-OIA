from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


IDENTITY_KEYS = (
    "run_id",
    "run_root",
    "git_head",
    "source_tree_hash",
    "split_manifest_hash",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_identity(
    run_root: str | Path,
    run_id: str,
    git_head: str,
    source_tree_hash: str,
    split_manifest_path: str | Path,
) -> dict[str, str]:
    if not run_id.strip():
        raise ValueError("run_id must be non-empty")
    return {
        "run_id": run_id,
        "run_root": str(Path(run_root).resolve()),
        "git_head": git_head,
        "source_tree_hash": source_tree_hash,
        "split_manifest_hash": sha256_file(split_manifest_path),
    }


def atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, destination)


def _identity_mismatches(actual: dict, expected: dict) -> list[str]:
    return [
        key
        for key in IDENTITY_KEYS
        if str(actual.get(key)) != str(expected.get(key))
    ]


def validate_stage_checkpoint(
    path: str | Path,
    expected_identity: dict,
    expected_stage: str,
    expected_parent_sha256: str | None = None,
) -> dict:
    checkpoint_path = Path(path).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    errors: list[str] = []
    if payload.get("stage") != expected_stage:
        errors.append(
            f"stage expected {expected_stage!r}, got {payload.get('stage')!r}"
        )
    actual_identity = payload.get("run_identity") or {}
    errors.extend(_identity_mismatches(actual_identity, expected_identity))
    manifest = payload.get("manifest") or {}
    if manifest.get("external_task_checkpoint"):
        errors.append("external task checkpoint is forbidden")
    if expected_parent_sha256 is not None and payload.get(
        "parent_checkpoint_sha256"
    ) != expected_parent_sha256:
        errors.append("parent checkpoint sha256 mismatch")
    if errors:
        raise RuntimeError(
            f"Stage checkpoint contract failed for {checkpoint_path}: "
            + ", ".join(errors)
        )
    return payload


def promote_stage_a_checkpoint(
    source_path: str | Path,
    destination_path: str | Path,
    identity: dict,
) -> dict[str, str]:
    source = Path(source_path).resolve()
    destination = Path(destination_path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    manifest = payload.get("manifest") or {}
    errors = []
    if manifest.get("external_task_checkpoint"):
        errors.append("external task checkpoint is forbidden")
    expected_manifest = {
        "git_head": identity["git_head"],
        "source_tree_hash": identity["source_tree_hash"],
        "split_manifest_hash": identity["split_manifest_hash"],
    }
    for key, expected in expected_manifest.items():
        if str(manifest.get(key)) != str(expected):
            errors.append(f"{key} mismatch")
    if payload.get("selection_split") != "train_audit":
        errors.append("Stage A checkpoint was not selected on train_audit")
    if errors:
        raise RuntimeError("Cannot promote Stage A checkpoint: " + ", ".join(errors))

    promoted = dict(payload)
    promoted.update(
        {
            "stage": "base_selected",
            "run_identity": dict(identity),
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": sha256_file(source),
        }
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(promoted, temporary)
    os.replace(temporary, destination)
    return {
        "checkpoint": str(destination),
        "checkpoint_sha256": sha256_file(destination),
        "source_checkpoint_sha256": promoted["source_checkpoint_sha256"],
    }
