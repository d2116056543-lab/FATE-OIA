from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ACTION_DIM = 4
REASON_DIM = 21
LABEL_DIM = ACTION_DIM + REASON_DIM
DEFAULT_SEED = 20260713


@dataclass(frozen=True)
class ICDORTrainSplits:
    train_core_indices: tuple[int, ...]
    train_audit_indices: tuple[int, ...]
    audit_visual_indices: tuple[int, ...]
    audit_target_indices: tuple[int, ...]
    train_calib_indices: tuple[int, ...]
    seed: int
    split_sha256: str
    file_names: tuple[str, ...]
    label_positive_counts: tuple[int, ...]
    audit_positive_counts: tuple[int, ...]
    audit_visual_positive_counts: tuple[int, ...]
    audit_target_positive_counts: tuple[int, ...]
    calib_positive_counts: tuple[int, ...]


def _value(sample: Any, key: str) -> Any:
    return sample.get(key) if isinstance(sample, dict) else getattr(sample, key, None)


def _metadata(dataset: Any) -> list[Any]:
    if not hasattr(dataset, "samples") or not isinstance(dataset.samples, (list, tuple)):
        raise ValueError("IC-DOR split requires ordered dataset.samples metadata")
    if len(dataset) != len(dataset.samples):
        raise ValueError("IC-DOR split dataset length must match metadata length")
    return list(dataset.samples)


def _label_vector(sample: Any) -> tuple[int, ...]:
    action = _value(sample, "action")
    reason = _value(sample, "reason")
    if not isinstance(action, (list, tuple)) or not isinstance(reason, (list, tuple)):
        raise ValueError("IC-DOR split metadata must contain action/reason sequences")
    if len(action) != ACTION_DIM or len(reason) != REASON_DIM:
        raise ValueError("IC-DOR split requires exactly 4 action and 21 reason labels")
    values = tuple(action) + tuple(reason)
    if any(float(value) not in {0.0, 1.0} for value in values):
        raise ValueError("IC-DOR split labels must be binary")
    return tuple(int(value) for value in values)


def _stable_rank(seed: int, file_name: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{file_name}".encode("utf-8")).digest(), "big")


def _select_stratified(
    labels: list[tuple[int, ...]],
    file_names: list[str],
    candidates: set[int],
    count: int,
    seed: int,
) -> list[int]:
    """Select one deterministic subset while prioritizing currently rare labels."""
    selected: list[int] = []
    selected_counts = [0] * LABEL_DIM
    total_counts = [sum(row[label] for row in labels) for label in range(LABEL_DIM)]
    ranked = {index: _stable_rank(seed, file_names[index]) for index in candidates}
    while len(selected) < count:
        remaining = candidates.difference(selected)
        if not remaining:
            raise RuntimeError("IC-DOR split selection exhausted candidates")
        scores: dict[int, float] = {}
        for index in remaining:
            score = 0.0
            for label, value in enumerate(labels[index]):
                if value:
                    expected = (len(selected) + 1) * total_counts[label] / max(len(labels), 1)
                    deficit = max(expected - selected_counts[label], 0.0)
                    score += 1.0 + deficit / max(total_counts[label], 1)
            scores[index] = score
        best_score = max(scores.values())
        tied = [index for index, score in scores.items() if score == best_score]
        chosen = min(tied, key=lambda index: (ranked[index], file_names[index]))
        selected.append(chosen)
        for label, value in enumerate(labels[chosen]):
            selected_counts[label] += value
    return sorted(selected)


def _canonical_payload(
    file_names: list[str], labels: list[tuple[int, ...]], seed: int, core: list[int], audit_visual: list[int],
    audit_target: list[int], calib: list[int]
) -> bytes:
    payload = {
        "seed": seed,
        "file_names": file_names,
        "labels": labels,
        "train_core_indices": core,
        "train_audit_indices": sorted(set(audit_visual) | set(audit_target)),
        "audit_visual_indices": audit_visual,
        "audit_target_indices": audit_target,
        "train_calib_indices": calib,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def make_icdor_train_splits(dataset: Any, *, seed: int = DEFAULT_SEED) -> ICDORTrainSplits:
    if type(seed) is not int:
        raise ValueError("IC-DOR split seed must be an exact integer")
    samples = _metadata(dataset)
    if len(samples) < 10:
        raise ValueError("IC-DOR split requires at least ten train samples")
    file_names: list[str] = []
    labels: list[tuple[int, ...]] = []
    for sample in samples:
        if _value(sample, "split") != "train":
            raise ValueError("IC-DOR split accepts train samples only")
        file_name = _value(sample, "file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("IC-DOR split requires non-empty file_name metadata")
        file_names.append(file_name)
        labels.append(_label_vector(sample))
    if len(set(file_names)) != len(file_names):
        raise ValueError("IC-DOR split requires unique file names")

    # Keep the two audit populations disjoint. Their union remains the
    # legacy train_audit view consumed by existing collectors.
    audit_visual_count = max(1, round(len(samples) * 0.05))
    audit_target_count = max(1, round(len(samples) * 0.05))
    calib_count = max(1, round(len(samples) * 0.10))
    if audit_visual_count + audit_target_count + calib_count >= len(samples):
        raise ValueError("IC-DOR split leaves no train_core samples")
    all_indices = set(range(len(samples)))
    audit_visual = _select_stratified(labels, file_names, all_indices, audit_visual_count, seed)
    audit_target = _select_stratified(
        labels, file_names, all_indices.difference(audit_visual), audit_target_count, seed + 1
    )
    audit = sorted(set(audit_visual) | set(audit_target))
    calib = _select_stratified(labels, file_names, all_indices.difference(audit), calib_count, seed + 2)
    core = sorted(all_indices.difference(audit).difference(calib))
    if set(core) & set(audit) or set(core) & set(calib) or set(audit_visual) & set(audit_target):
        raise RuntimeError("IC-DOR split must be pairwise disjoint")
    if set(core) | set(audit) | set(calib) != all_indices:
        raise RuntimeError("IC-DOR split union must equal original train IDs")

    positive_counts = tuple(sum(row[label] for row in labels) for label in range(LABEL_DIM))
    audit_counts = tuple(sum(labels[index][label] for index in audit) for label in range(LABEL_DIM))
    audit_visual_counts = tuple(sum(labels[index][label] for index in audit_visual) for label in range(LABEL_DIM))
    audit_target_counts = tuple(sum(labels[index][label] for index in audit_target) for label in range(LABEL_DIM))
    calib_counts = tuple(sum(labels[index][label] for index in calib) for label in range(LABEL_DIM))
    split_hash = hashlib.sha256(
        _canonical_payload(file_names, labels, seed, core, audit_visual, audit_target, calib)
    ).hexdigest().upper()
    return ICDORTrainSplits(
        train_core_indices=tuple(core),
        train_audit_indices=tuple(audit),
        audit_visual_indices=tuple(sorted(audit_visual)),
        audit_target_indices=tuple(sorted(audit_target)),
        train_calib_indices=tuple(calib),
        seed=seed,
        split_sha256=split_hash,
        file_names=tuple(file_names),
        label_positive_counts=positive_counts,
        audit_positive_counts=audit_counts,
        audit_visual_positive_counts=audit_visual_counts,
        audit_target_positive_counts=audit_target_counts,
        calib_positive_counts=calib_counts,
    )


def write_icdor_split_manifest(result: ICDORTrainSplits, output_path: str | Path) -> None:
    payload = {
        "seed": result.seed,
        "split_sha256": result.split_sha256,
        "file_names": list(result.file_names),
        "train_core_indices": list(result.train_core_indices),
        "train_audit_indices": list(result.train_audit_indices),
        "audit_visual_indices": list(result.audit_visual_indices),
        "audit_target_indices": list(result.audit_target_indices),
        "train_calib_indices": list(result.train_calib_indices),
        "labels": [
            {
                "label_index": label,
                "label_type": "action" if label < ACTION_DIM else "reason",
                "local_index": label if label < ACTION_DIM else label - ACTION_DIM,
                "positive_count": result.label_positive_counts[label],
                "audit_positive_count": result.audit_positive_counts[label],
                "audit_visual_positive_count": result.audit_visual_positive_counts[label],
                "audit_target_positive_count": result.audit_target_positive_counts[label],
                "calib_positive_count": result.calib_positive_counts[label],
            }
            for label in range(LABEL_DIM)
        ],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
