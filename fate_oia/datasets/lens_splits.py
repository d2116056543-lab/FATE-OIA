from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch


def _digest(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def make_lens_splits(file_names: list[str], labels: torch.Tensor, *, seed: int = 3407) -> dict[str, list[int] | str]:
    """Deterministic greedy multi-label iterative stratification of train only."""
    if len(file_names) != labels.shape[0]:
        raise ValueError("file_names/labels mismatch")
    n = len(file_names)
    names = ("train_main", "train_audit", "train_calib")
    capacities = torch.tensor([n - 2 * round(n * 0.05), round(n * 0.05), round(n * 0.05)], dtype=torch.long)
    capacities[0] += n - int(capacities.sum())
    label_counts = labels.float().sum(0)
    fractions = capacities.float() / max(n, 1)
    desired = fractions[:, None] * label_counts[None, :]
    # A label with enough examples must remain auditable in every train-only split.
    eligible = label_counts >= len(names)
    desired[1:, eligible] = torch.maximum(desired[1:, eligible], torch.ones_like(desired[1:, eligible]))
    desired[0, eligible] = torch.maximum(label_counts[eligible] - desired[1, eligible] - desired[2, eligible], torch.ones_like(label_counts[eligible]))
    remaining_capacity = capacities.clone()
    remaining_desired = desired.clone()
    generator = torch.Generator().manual_seed(seed)
    tie = torch.rand(n, generator=generator) * 1e-6
    rarity = (labels.float() / label_counts.clamp_min(1.0)).sum(-1)
    bucket_lists: list[list[int]] = [[], [], []]
    assigned: set[int] = set()
    # Reserve auditable examples for rare labels before capacity is consumed by common labels.
    for label in torch.argsort(label_counts).tolist():
        if label_counts[label] < len(names):
            continue
        positives = torch.where(labels[:, label] > 0.5)[0].tolist()
        for target in (1, 2):
            if any(labels[index, label] > 0.5 for index in bucket_lists[target]):
                continue
            candidates = [index for index in positives if index not in assigned]
            if not candidates or remaining_capacity[target] <= 0:
                continue
            index = max(candidates, key=lambda item: float(rarity[item] + tie[item]))
            bucket_lists[target].append(index)
            assigned.add(index)
            remaining_capacity[target] -= 1
            active = labels[index].bool()
            remaining_desired[target, active] = (remaining_desired[target, active] - 1.0).clamp_min(0.0)
    ordered = [index for index in torch.argsort(rarity + tie, descending=True).tolist() if index not in assigned]
    for index in ordered:
        active = labels[index].bool()
        candidate = remaining_capacity > 0
        score = remaining_capacity.float() / capacities.clamp_min(1).float()
        if active.any():
            score = score + (remaining_desired[:, active] / label_counts[active].clamp_min(1.0)).sum(-1)
        score[~candidate] = -float("inf")
        target = int(torch.argmax(score))
        bucket_lists[target].append(index)
        remaining_capacity[target] -= 1
        if active.any():
            remaining_desired[target, active] = (remaining_desired[target, active] - 1.0).clamp_min(0.0)
    buckets = {name: bucket_lists[i] for i, name in enumerate(names)}
    manifest: dict[str, list[int] | str] = dict(buckets)
    manifest["all_file_names_sha256"] = _digest(file_names)
    return manifest


def write_split_manifest(path: str | Path, manifest: dict[str, list[int] | str], file_names: list[str]) -> None:
    path=Path(path)
    output = dict(manifest)
    for key in ("train_main", "train_audit", "train_calib"):
        output[f"{key}_ids"] = [file_names[i] for i in output[key]]  # type: ignore[index]
        output[f"{key}_sha256"] = _digest(output[f"{key}_ids"])
        (path.parent/f"{key}_ids.json").write_text(json.dumps(output[f"{key}_ids"],ensure_ascii=False,indent=2),encoding="utf-8")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
