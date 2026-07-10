from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_CALIB_FRACTION = 0.10
DEFAULT_SPLIT_SEED = 20260710
DEFAULT_MIN_CALIB_POSITIVES = 20
_ACTION_DIM = 4
_REASON_DIM = 21
_LABEL_DIM = _ACTION_DIM + _REASON_DIM


def _sample_value(sample: Any, key: str) -> Any:
    if isinstance(sample, dict):
        return sample.get(key)
    return getattr(sample, key, None)


def _metadata_samples(dataset: Any) -> list[tuple[int, Any]]:
    if hasattr(dataset, "samples"):
        samples = dataset.samples
        if not isinstance(samples, (list, tuple)):
            raise ValueError("dataset.samples must be an ordered sequence")
        return list(enumerate(samples))
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices") and hasattr(dataset.dataset, "samples"):
        base_samples = dataset.dataset.samples
        return [(local_index, base_samples[int(base_index)]) for local_index, base_index in enumerate(dataset.indices)]
    raise ValueError("MOSAIC split requires dataset.samples metadata and must not decode images")


def _binary_label_vector(sample: Any) -> tuple[int, ...]:
    action = _sample_value(sample, "action")
    reason = _sample_value(sample, "reason")
    if not isinstance(action, (list, tuple)) or not isinstance(reason, (list, tuple)):
        raise ValueError("MOSAIC split metadata must expose action/reason sequences")
    if len(action) != _ACTION_DIM or len(reason) != _REASON_DIM:
        raise ValueError("MOSAIC split requires exactly 4 action and 21 reason labels")
    values = tuple(action) + tuple(reason)
    normalized: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("MOSAIC split labels must be finite binary numbers")
        if float(value) not in {0.0, 1.0}:
            raise ValueError("MOSAIC split labels must be binary")
        normalized.append(int(value))
    return tuple(normalized)


def _stable_rank(seed: int, file_name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{file_name}".encode("utf-8")).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _split_payload(
    *,
    seed: int,
    calib_fraction: float,
    file_names: list[str],
    label_sha256: str,
    min_calib_positives: int,
    main_indices: list[int],
    calib_indices: list[int],
) -> dict[str, Any]:
    return {
        "seed": seed,
        "calib_fraction": calib_fraction,
        "file_names": file_names,
        "label_sha256": label_sha256,
        "min_calib_positives": min_calib_positives,
        "train_main_indices": main_indices,
        "train_calib_indices": calib_indices,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_multilabel_train_calib_indices(
    dataset: Any,
    calib_fraction: float = DEFAULT_CALIB_FRACTION,
    seed: int = DEFAULT_SPLIT_SEED,
    min_calib_positives: int = DEFAULT_MIN_CALIB_POSITIVES,
    output_dir: str | Path | None = None,
) -> tuple[list[int], list[int]]:
    if isinstance(calib_fraction, bool) or not isinstance(calib_fraction, (int, float)):
        raise ValueError("calib_fraction must be numeric")
    calib_fraction = float(calib_fraction)
    if not 0.0 < calib_fraction < 1.0:
        raise ValueError("calib_fraction must be between 0 and 1")
    if type(seed) is not int or type(min_calib_positives) is not int or min_calib_positives < 0:
        raise ValueError("seed/min_calib_positives must be exact non-negative integers")

    metadata = _metadata_samples(dataset)
    sample_count = len(metadata)
    if len(dataset) != sample_count:
        raise ValueError("dataset length does not match metadata length")
    if sample_count < 2:
        raise ValueError("MOSAIC train/calib split requires at least two samples")

    file_names: list[str] = []
    labels: list[tuple[int, ...]] = []
    for _, sample in metadata:
        split = _sample_value(sample, "split")
        if split != "train":
            raise ValueError("MOSAIC train/calib split accepts train samples only")
        file_name = _sample_value(sample, "file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("MOSAIC split requires non-empty file_name metadata")
        file_names.append(file_name)
        labels.append(_binary_label_vector(sample))
    if len(set(file_names)) != sample_count:
        raise ValueError("MOSAIC split found duplicate file_name metadata")

    calib_count = min(sample_count - 1, max(1, int(round(sample_count * calib_fraction))))
    available = [sum(row[label] for row in labels) for label in range(_LABEL_DIM)]
    ratio_targets = [count * calib_fraction for count in available]
    minimum_targets = [min(min_calib_positives, count) if count else 0 for count in available]
    targets = [min(count, max(int(round(count * calib_fraction)), minimum)) for count, minimum in zip(available, minimum_targets)]
    ranks = [_stable_rank(seed, file_name) for file_name in file_names]
    label_sha256 = hashlib.sha256(_canonical_json_bytes(labels)).hexdigest().upper()

    selected: list[int] = []
    selected_set: set[int] = set()
    current = [0] * _LABEL_DIM
    while len(selected) < calib_count:
        deficits = [max(targets[label] - current[label], 0) for label in range(_LABEL_DIM)]
        best_index: int | None = None
        best_key: tuple[float, float, int] | None = None
        for index, row in enumerate(labels):
            if index in selected_set:
                continue
            coverage_gain = sum(
                row[label] * deficits[label] / max(targets[label], 1)
                for label in range(_LABEL_DIM)
                if deficits[label] > 0
            )
            ratio_gain = sum(
                (current[label] - ratio_targets[label]) ** 2
                - (current[label] + row[label] - ratio_targets[label]) ** 2
                for label in range(_LABEL_DIM)
            )
            key = (coverage_gain, ratio_gain, -ranks[index])
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            raise RuntimeError("MOSAIC split selection exhausted candidates")
        selected.append(best_index)
        selected_set.add(best_index)
        for label in range(_LABEL_DIM):
            current[label] += labels[best_index][label]

    calib_indices = sorted(selected)
    main_indices = [index for index in range(sample_count) if index not in selected_set]
    main_counts = [available[label] - current[label] for label in range(_LABEL_DIM)]
    label_stats = [
        {
            "label_index": label,
            "label_type": "action" if label < _ACTION_DIM else "reason",
            "local_index": label if label < _ACTION_DIM else label - _ACTION_DIM,
            "available_positive": available[label],
            "target_calib_positive": targets[label],
            "calib_positive": current[label],
            "main_positive": main_counts[label],
            "target_met": current[label] >= targets[label],
            "shortfall": max(targets[label] - current[label], 0),
        }
        for label in range(_LABEL_DIM)
    ]

    if output_dir is not None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        stats = {
            "seed": seed,
            "calib_fraction": calib_fraction,
            "sample_count": sample_count,
            "main_count": len(main_indices),
            "calib_count": len(calib_indices),
            "label_count": _LABEL_DIM,
            "min_calib_positives": min_calib_positives,
            "all_targets_met": all(row["target_met"] for row in label_stats),
            "labels": label_stats,
        }
        payload = _split_payload(
            seed=seed,
            calib_fraction=calib_fraction,
            file_names=file_names,
            label_sha256=label_sha256,
            min_calib_positives=min_calib_positives,
            main_indices=main_indices,
            calib_indices=calib_indices,
        )
        split_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest().upper()
        stats_hash = hashlib.sha256(_canonical_json_bytes(stats)).hexdigest().upper()
        _write_json(output_path / "train_main_indices.json", main_indices)
        _write_json(output_path / "train_calib_indices.json", calib_indices)
        _write_json(output_path / "train_calib_split_stats.json", stats)
        _write_json(
            output_path / "train_calib_split_hash.json",
            {
                "algorithm": "SHA256",
                "split_sha256": split_hash,
                "seed": seed,
                "calib_fraction": calib_fraction,
                "label_sha256": label_sha256,
                "stats_sha256": stats_hash,
                "min_calib_positives": min_calib_positives,
            },
        )
    return main_indices, calib_indices


def verify_train_calib_split_artifacts(output_dir: str | Path, dataset: Any) -> bool:
    output_path = Path(output_dir)
    main_indices = json.loads((output_path / "train_main_indices.json").read_text(encoding="utf-8"))
    calib_indices = json.loads((output_path / "train_calib_indices.json").read_text(encoding="utf-8"))
    stats = json.loads((output_path / "train_calib_split_stats.json").read_text(encoding="utf-8"))
    hash_record = json.loads((output_path / "train_calib_split_hash.json").read_text(encoding="utf-8"))
    stats_hash = hashlib.sha256(_canonical_json_bytes(stats)).hexdigest().upper()
    if hash_record.get("stats_sha256") != stats_hash:
        raise ValueError("split stats hash mismatch")
    metadata = _metadata_samples(dataset)
    if len(dataset) != len(metadata):
        raise ValueError("dataset length does not match metadata length")
    file_names = [_sample_value(sample, "file_name") for _, sample in metadata]
    labels = [_binary_label_vector(sample) for _, sample in metadata]
    label_sha256 = hashlib.sha256(_canonical_json_bytes(labels)).hexdigest().upper()
    payload = _split_payload(
        seed=stats["seed"],
        calib_fraction=stats["calib_fraction"],
        file_names=file_names,
        label_sha256=label_sha256,
        min_calib_positives=stats["min_calib_positives"],
        main_indices=main_indices,
        calib_indices=calib_indices,
    )
    actual_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest().upper()
    if (
        hash_record.get("algorithm") != "SHA256"
        or hash_record.get("split_sha256") != actual_hash
        or hash_record.get("label_sha256") != label_sha256
        or hash_record.get("min_calib_positives") != stats.get("min_calib_positives")
    ):
        raise ValueError("split artifact hash mismatch")
    if any(type(index) is not int for index in main_indices + calib_indices):
        raise ValueError("split artifacts contain non-integer indices")
    if main_indices != sorted(main_indices) or calib_indices != sorted(calib_indices):
        raise ValueError("split artifact indices must be sorted")
    if set(main_indices) & set(calib_indices) or sorted(main_indices + calib_indices) != list(range(len(metadata))):
        raise ValueError("split artifacts do not form a complete disjoint partition")
    if stats.get("sample_count") != len(metadata) or stats.get("main_count") != len(main_indices) or stats.get(
        "calib_count"
    ) != len(calib_indices):
        raise ValueError("split artifact stats do not match indices")
    return True
