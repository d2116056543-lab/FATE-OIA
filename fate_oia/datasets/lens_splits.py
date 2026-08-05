from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def make_lens_splits(file_names: list[str], labels: torch.Tensor, *, seed: int = 3407) -> dict[str, list[int] | str]:
    """Deterministic label-balanced round-robin assignment without test/val involvement."""
    if len(file_names) != labels.shape[0]:
        raise ValueError("file_names/labels mismatch")
    generator = torch.Generator().manual_seed(seed)
    score = labels.float().sum(-1) + torch.rand(len(file_names), generator=generator) * 1e-3
    ordered = torch.argsort(score, descending=True).tolist()
    buckets = {"train_main": [], "train_audit": [], "train_calib": []}
    cycle = ["train_main"] * 18 + ["train_audit"] + ["train_calib"]
    for rank, index in enumerate(ordered):
        buckets[cycle[rank % len(cycle)]].append(index)
    manifest: dict[str, list[int] | str] = dict(buckets)
    manifest["all_file_names_sha256"] = _digest(file_names)
    return manifest


def write_split_manifest(path: str | Path, manifest: dict[str, list[int] | str], file_names: list[str]) -> None:
    output = dict(manifest)
    for key in ("train_main", "train_audit", "train_calib"):
        output[f"{key}_ids"] = [file_names[i] for i in output[key]]  # type: ignore[index]
        output[f"{key}_sha256"] = _digest(output[f"{key}_ids"])
    Path(path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
