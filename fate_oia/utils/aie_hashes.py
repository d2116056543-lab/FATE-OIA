from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def aie_source_tree_sha256(root: str | Path = ".") -> str:
    base = Path(root)
    paths = list((base / "fate_oia").rglob("aie_*.py")) + list((base / "tests").glob("test_aie_*.py"))
    paths += [
        base / "configs/aie_scene_predicates.yaml",
        base / "configs/aie_reason_counter_evidence.yaml",
        base / "configs/fate_oia_train_360x640_aie_oia_v1.yaml",
        base / "scripts/FATE_OIA_aie_oia_v1_pilot.ps1",
        base / "scripts/FATE_OIA_aie_oia_v1_foreground.ps1",
    ]
    manifest = {path.relative_to(base).as_posix(): file_sha256(path) for path in sorted(set(paths)) if path.exists()}
    return object_sha256(manifest)

