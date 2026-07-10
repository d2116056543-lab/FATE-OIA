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
    if hasattr(dataset, "dataset") and hasattr(dataset, "indices") and hasattr(dataset.dataset, "samples"):
        base_samples = dataset.dataset.samples
        indices = list(dataset.indices)
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("subset indices must be exact non-negative integers")
        if any(index >= len(base_samples) for index in indices):
            raise ValueError("subset indices exceed base dataset metadata")
        if len(set(indices)) != len(indices):
            raise ValueError("subset indices must be unique")
        return [(local_index, base_samples[base_index]) for local_index, base_index in enumerate(indices)]
    if hasattr(dataset, "samples"):
        samples = dataset.samples
        if not isinstance(samples, (list, tuple)):
            raise ValueError("dataset.samples must be an ordered sequence")
        return list(enumerate(samples))
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


def _repair_coverage_with_milp(
    *,
    labels: list[tuple[int, ...]],
    targets: list[int],
    calib_count: int,
    greedy_selected: list[int],
    ranks: list[int],
    file_names: list[str],
) -> tuple[list[int] | None, str]:
    """Find an exact-cardinality coverage solution when the deterministic greedy pass misses one."""
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as exc:
        raise RuntimeError("scipy.optimize.milp is required for exact MOSAIC split repair") from exc

    sample_count = len(labels)
    canonical_order = sorted(range(sample_count), key=lambda index: (ranks[index], file_names[index]))
    canonical_labels = [labels[index] for index in canonical_order]
    greedy_set = set(greedy_selected)
    tie_scale = 1.0 / max(sample_count * sample_count * 4.0, 1.0)
    objective = np.asarray(
        [
            (-1.0 if original_index in greedy_set else 0.0) + tie_scale * canonical_index
            for canonical_index, original_index in enumerate(canonical_order)
        ],
        dtype=np.float64,
    )

    rows = [np.ones(sample_count, dtype=np.float64)]
    lower = [float(calib_count)]
    upper = [float(calib_count)]
    for label, target in enumerate(targets):
        if target <= 0:
            continue
        rows.append(np.asarray([sample_labels[label] for sample_labels in canonical_labels], dtype=np.float64))
        lower.append(float(target))
        upper.append(np.inf)

    result = milp(
        c=objective,
        integrality=np.ones(sample_count, dtype=np.int8),
        bounds=Bounds(np.zeros(sample_count), np.ones(sample_count)),
        constraints=LinearConstraint(np.stack(rows), np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 120.0, "presolve": True},
    )
    if result.status == 2:
        return None, "scipy_milp_proved_infeasible"
    if not result.success or result.x is None:
        raise RuntimeError(f"MOSAIC exact split repair failed: status={result.status}, message={result.message}")

    selected = sorted(canonical_order[index] for index, value in enumerate(result.x) if value >= 0.5)
    if len(selected) != calib_count:
        raise RuntimeError("MOSAIC exact split repair returned the wrong calibration cardinality")
    repaired_counts = [sum(labels[index][label] for index in selected) for label in range(_LABEL_DIM)]
    if any(repaired_counts[label] < targets[label] for label in range(_LABEL_DIM)):
        raise RuntimeError("MOSAIC exact split repair returned a coverage-invalid solution")
    return selected, "scipy_milp_exact_repair"


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
    if calib_fraction != DEFAULT_CALIB_FRACTION:
        raise ValueError("MOSAIC protocol requires calib_fraction=0.10")
    if type(seed) is not int or type(min_calib_positives) is not int or min_calib_positives < 0:
        raise ValueError("seed/min_calib_positives must be exact non-negative integers")
    if seed != DEFAULT_SPLIT_SEED:
        raise ValueError("MOSAIC protocol requires seed=20260710")
    if min_calib_positives != DEFAULT_MIN_CALIB_POSITIVES:
        raise ValueError("MOSAIC protocol requires min_calib_positives=20")

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

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for deterministic MOSAIC split scoring") from exc
    label_matrix = np.asarray(labels, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    ratio_target_array = np.asarray(ratio_targets, dtype=np.float64)
    selected: list[int] = []
    selected_mask = np.zeros(sample_count, dtype=np.bool_)
    current_array = np.zeros(_LABEL_DIM, dtype=np.float64)
    while len(selected) < calib_count:
        available_mask = ~selected_mask
        if not available_mask.any():
            raise RuntimeError("MOSAIC split selection exhausted candidates")
        deficits = np.maximum(target_array - current_array, 0.0)
        coverage_weights = np.divide(
            deficits,
            np.maximum(target_array, 1.0),
            out=np.zeros_like(deficits),
            where=deficits > 0,
        )
        coverage_gain = label_matrix @ coverage_weights
        ratio_coefficients = -2.0 * (current_array - ratio_target_array) - 1.0
        ratio_gain = label_matrix @ ratio_coefficients
        max_coverage = coverage_gain[available_mask].max()
        candidates = np.flatnonzero(available_mask & (coverage_gain == max_coverage))
        max_ratio = ratio_gain[candidates].max()
        candidates = candidates[ratio_gain[candidates] == max_ratio]
        best_index = min((int(index) for index in candidates), key=lambda index: (ranks[index], file_names[index]))
        selected.append(best_index)
        selected_mask[best_index] = True
        current_array += label_matrix[best_index]

    selected_set = set(selected)
    current = [int(value) for value in current_array.tolist()]

    coverage_solver = "deterministic_greedy"
    if any(current[label] < targets[label] for label in range(_LABEL_DIM)):
        repaired, coverage_solver = _repair_coverage_with_milp(
            labels=labels,
            targets=targets,
            calib_count=calib_count,
            greedy_selected=selected,
            ranks=ranks,
            file_names=file_names,
        )
        if repaired is not None:
            selected = repaired
            selected_set = set(selected)
            current = [sum(labels[index][label] for index in selected) for label in range(_LABEL_DIM)]

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
            "coverage_solver": coverage_solver,
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
    file_names: list[str] = []
    labels: list[tuple[int, ...]] = []
    for _, sample in metadata:
        if _sample_value(sample, "split") != "train":
            raise ValueError("MOSAIC train/calib split accepts train samples only")
        file_name = _sample_value(sample, "file_name")
        if not isinstance(file_name, str) or not file_name.strip():
            raise ValueError("MOSAIC split requires non-empty file_name metadata")
        file_names.append(file_name)
        labels.append(_binary_label_vector(sample))
    if len(set(file_names)) != len(file_names):
        raise ValueError("MOSAIC split found duplicate file_name metadata")

    protocol_metadata = {
        "seed": DEFAULT_SPLIT_SEED,
        "calib_fraction": DEFAULT_CALIB_FRACTION,
        "min_calib_positives": DEFAULT_MIN_CALIB_POSITIVES,
    }
    if any(stats.get(key) != value or hash_record.get(key) != value for key, value in protocol_metadata.items()):
        raise ValueError("split hash metadata mismatch")
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
    ):
        raise ValueError("split artifact hash mismatch")
    if any(type(index) is not int for index in main_indices + calib_indices):
        raise ValueError("split artifacts contain non-integer indices")
    if main_indices != sorted(main_indices) or calib_indices != sorted(calib_indices):
        raise ValueError("split artifact indices must be sorted")
    if set(main_indices) & set(calib_indices) or sorted(main_indices + calib_indices) != list(range(len(metadata))):
        raise ValueError("split artifacts do not form a complete disjoint partition")
    expected_main, expected_calib = make_multilabel_train_calib_indices(dataset)
    if main_indices != expected_main or calib_indices != expected_calib:
        raise ValueError("split artifacts do not match the deterministic protocol partition")

    available = [sum(row[label] for row in labels) for label in range(_LABEL_DIM)]
    expected_calib_count = min(
        len(metadata) - 1,
        max(1, int(round(len(metadata) * DEFAULT_CALIB_FRACTION))),
    )
    targets = [
        min(
            count,
            max(
                int(round(count * DEFAULT_CALIB_FRACTION)),
                min(DEFAULT_MIN_CALIB_POSITIVES, count) if count else 0,
            ),
        )
        for count in available
    ]
    calib_counts = [sum(labels[index][label] for index in calib_indices) for label in range(_LABEL_DIM)]
    expected_label_stats = [
        {
            "label_index": label,
            "label_type": "action" if label < _ACTION_DIM else "reason",
            "local_index": label if label < _ACTION_DIM else label - _ACTION_DIM,
            "available_positive": available[label],
            "target_calib_positive": targets[label],
            "calib_positive": calib_counts[label],
            "main_positive": available[label] - calib_counts[label],
            "target_met": calib_counts[label] >= targets[label],
            "shortfall": max(targets[label] - calib_counts[label], 0),
        }
        for label in range(_LABEL_DIM)
    ]
    if stats.get("sample_count") != len(metadata) or stats.get("main_count") != len(main_indices) or stats.get(
        "calib_count"
    ) != len(calib_indices):
        raise ValueError("split artifact stats do not match indices")
    if len(calib_indices) != expected_calib_count:
        raise ValueError("split artifact calibration cardinality violates the fixed protocol")
    if stats.get("label_count") != _LABEL_DIM or stats.get("labels") != expected_label_stats:
        raise ValueError("split artifact per-label stats do not match dataset labels")
    if stats.get("all_targets_met") is not all(row["target_met"] for row in expected_label_stats):
        raise ValueError("split artifact target status does not match recomputed coverage")
    if stats.get("coverage_solver") not in {
        "deterministic_greedy",
        "scipy_milp_exact_repair",
        "scipy_milp_proved_infeasible",
    }:
        raise ValueError("split artifact has an invalid coverage solver status")
    return True
