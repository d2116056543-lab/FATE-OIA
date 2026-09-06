from __future__ import annotations

import os
import random
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity(config_path: str | Path, data_audit_path: str | Path | None = None) -> dict[str, Any]:
    identity = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_tree": subprocess.check_output(["git", "write-tree"], text=True).strip(),
        "config_sha256": file_sha256(config_path),
    }
    if data_audit_path is not None and Path(data_audit_path).is_file():
        identity["data_audit_sha256"] = file_sha256(data_audit_path)
    return identity


def verify_resume_identity(saved: dict[str, Any], current: dict[str, Any]) -> None:
    mismatches = {key: (saved.get(key), value) for key, value in current.items() if saved.get(key) != value}
    if mismatches:
        raise RuntimeError(f"resume identity mismatch: {mismatches}")


def capture_rng() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"): torch.cuda.set_rng_state_all(state["cuda"])


def atomic_torch_save(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temp)
    with temp.open("ab") as handle:
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, target)
