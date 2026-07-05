from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTION_NAMES = ("maintain_speed", "reduce_speed", "stop_car")
SPLITS = ("train", "val", "test")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dump_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def _load_records(path: Path) -> list[dict[str, Any]]:
    obj = _load_pickle(path)
    if isinstance(obj, dict):
        for key in ("samples", "records", "data"):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, list):
        raise TypeError(f"Expected list records in {path}, got {type(obj)!r}")
    return [dict(row) for row in obj]


def _load_exp29(path: Path, expected_len: int) -> tuple[list[Any], list[Any]]:
    if not path.exists():
        return [[0.0] * 29 for _ in range(expected_len)], [0.0 for _ in range(expected_len)]
    obj = _load_pickle(path)
    if isinstance(obj, dict) and "labels" in obj:
        labels = list(obj["labels"])
        masks = list(obj.get("masks", [1.0] * len(labels)))
    elif isinstance(obj, list):
        labels = [row.get("exp29", row.get("labels", [0.0] * 29)) for row in obj]
        masks = [row.get("exp29_mask", row.get("mask", 1.0)) for row in obj]
    else:
        raise TypeError(f"Expected exp29 dict/list in {path}, got {type(obj)!r}")
    if len(labels) != expected_len:
        raise ValueError(f"Exp29 length mismatch for {path}: samples={expected_len}, labels={len(labels)}")
    return labels, masks


def decision_event_key(row: dict[str, Any]) -> str:
    video = str(row.get("video_id", ""))
    decision = row.get("decision_keyframe", row.get("decision_frame", row.get("decision_kf")))
    if video == "" or decision is None:
        raise ValueError(f"Missing video_id/decision_keyframe in row: {row}")
    return f"{video}::{int(decision)}"


def action_id(row: dict[str, Any]) -> int:
    soft = row.get("action_soft_target") or row.get("action_soft") or row.get("action_distribution")
    if soft is not None:
        return int(max(range(min(3, len(soft))), key=lambda idx: float(soft[idx])))
    name = str(row.get("action_name", ""))
    if name in ACTION_NAMES:
        return ACTION_NAMES.index(name)
    return int(row.get("action_label", row.get("action_majority", 0)))


def target_decision_gap(row: dict[str, Any]) -> int | None:
    target = row.get("target_frame")
    decision = row.get("decision_keyframe", row.get("decision_frame", row.get("decision_kf")))
    if target is None or decision is None:
        return None
    try:
        return int(target) - int(decision)
    except (TypeError, ValueError):
        return None


def _load_combined_package(package_root: Path) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    for split in SPLITS:
        sample_path = package_root / "samples" / f"{split}.pkl"
        if not sample_path.exists():
            continue
        records = _load_records(sample_path)
        labels, masks = _load_exp29(package_root / "reason_exp29" / f"{split}.pkl", len(records))
        for idx, record in enumerate(records):
            row = dict(record)
            row["_source_split"] = split
            row["_source_index"] = idx
            row["_exp29"] = labels[idx]
            row["_exp29_mask"] = masks[idx]
            combined.append(row)
    if not combined:
        raise FileNotFoundError(f"No PSI records found under {package_root / 'samples'}")
    return combined


def _split_events_by_action(events: dict[str, list[dict[str, Any]]], *, num_folds: int, seed: int) -> list[list[str]]:
    by_action: dict[int, list[str]] = defaultdict(list)
    for key, rows in events.items():
        by_action[action_id(rows[0])].append(key)
    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(num_folds)]
    fold_sizes = [0 for _ in range(num_folds)]
    for action in sorted(by_action):
        keys = list(by_action[action])
        rng.shuffle(keys)
        for key in keys:
            target_fold = min(range(num_folds), key=lambda idx: (fold_sizes[idx], len(folds[idx])))
            folds[target_fold].append(key)
            fold_sizes[target_fold] += len(events[key])
    return folds


def _take_dev_events(
    train_event_keys: list[str],
    events: dict[str, list[dict[str, Any]]],
    *,
    dev_fraction: float,
    seed: int,
) -> set[str]:
    if dev_fraction <= 0 or len(train_event_keys) <= 1:
        return set()
    by_action: dict[int, list[str]] = defaultdict(list)
    for key in train_event_keys:
        by_action[action_id(events[key][0])].append(key)
    rng = random.Random(seed + 9973)
    dev: set[str] = set()
    target_total = max(1, int(round(len(train_event_keys) * dev_fraction)))
    for action in sorted(by_action):
        keys = list(by_action[action])
        rng.shuffle(keys)
        take = int(round(len(keys) * dev_fraction))
        if len(keys) > 1:
            take = max(1, min(take, len(keys) - 1))
        else:
            take = 0
        dev.update(keys[:take])
    if len(dev) > target_total:
        ordered = sorted(dev)
        rng.shuffle(ordered)
        dev = set(ordered[:target_total])
    if not dev and train_event_keys:
        dev.add(train_event_keys[0])
    return dev


def _write_split(output_root: Path, split: str, event_keys: set[str], events: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    labels: list[Any] = []
    masks: list[Any] = []
    event_action_counts: Counter[int] = Counter()
    for key in sorted(event_keys):
        event_rows = events[key]
        event_action_counts[action_id(event_rows[0])] += 1
        for row in event_rows:
            clean = dict(row)
            labels.append(clean.pop("_exp29"))
            masks.append(clean.pop("_exp29_mask"))
            rows.append(clean)
    _dump_pickle(output_root / "samples" / f"{split}.pkl", rows)
    _dump_pickle(output_root / "reason_exp29" / f"{split}.pkl", {"labels": labels, "masks": masks})
    row_action_counts = Counter(action_id(row) for row in rows)
    gaps = [gap for row in rows if (gap := target_decision_gap(row)) is not None]
    return {
        "rows": len(rows),
        "events": len(event_keys),
        "row_action_counts": {ACTION_NAMES[idx]: row_action_counts.get(idx, 0) for idx in range(3)},
        "event_action_counts": {ACTION_NAMES[idx]: event_action_counts.get(idx, 0) for idx in range(3)},
        "target_decision_gap_min": min(gaps) if gaps else None,
        "target_decision_gap_max": max(gaps) if gaps else None,
        "target_decision_gap_mean": (sum(gaps) / len(gaps)) if gaps else None,
    }


def _shared_events(split_events: dict[str, set[str]]) -> int:
    total = 0
    names = list(split_events)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            total += len(split_events[left] & split_events[right])
    return total


def build_event_disjoint_package(
    package_root: str | Path,
    output_root: str | Path,
    *,
    fold: int = 0,
    num_folds: int = 5,
    dev_fraction: float = 0.15,
    seed: int = 20260706,
    max_target_decision_gap: int | None = None,
) -> dict[str, Any]:
    """Create a PSI package whose train/val/test splits are disjoint by decision event.

    The source package may already contain train/val/test splits, but those splits are
    treated only as storage shards. The new package groups all expanded rows by
    ``video_id::decision_keyframe`` and keeps every row from an event in exactly one
    output split.
    """
    source = Path(package_root)
    output = Path(output_root)
    if num_folds < 2:
        raise ValueError("num_folds must be >= 2")
    if fold < 0 or fold >= num_folds:
        raise ValueError(f"fold must be in [0, {num_folds}), got {fold}")

    source_rows = _load_combined_package(source)
    if max_target_decision_gap is not None:
        if max_target_decision_gap < 0:
            raise ValueError("max_target_decision_gap must be non-negative or None")
        combined = [
            row for row in source_rows
            if (gap := target_decision_gap(row)) is not None and 0 <= gap <= max_target_decision_gap
        ]
    else:
        combined = source_rows
    if not combined:
        raise ValueError(
            f"No PSI rows remain after max_target_decision_gap={max_target_decision_gap} filter"
        )
    events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in combined:
        events[decision_event_key(row)].append(row)
    folds = _split_events_by_action(events, num_folds=num_folds, seed=seed)
    test_events = set(folds[fold])
    train_candidates = [key for idx, keys in enumerate(folds) if idx != fold for key in keys]
    dev_events = _take_dev_events(train_candidates, events, dev_fraction=dev_fraction, seed=seed + fold)
    train_events = set(train_candidates) - dev_events
    split_events = {"train": train_events, "val": dev_events, "test": test_events}

    output.mkdir(parents=True, exist_ok=True)
    split_summary = {
        split: _write_split(output, split, keys, events)
        for split, keys in split_events.items()
    }
    summary = {
        "protocol": "psi_event_disjoint_gap_filtered_v1" if max_target_decision_gap is not None else "psi_event_disjoint_allrows_v1",
        "source_root": str(source),
        "output_root": str(output),
        "fold": fold,
        "num_folds": num_folds,
        "seed": seed,
        "dev_fraction": dev_fraction,
        "max_target_decision_gap": max_target_decision_gap,
        "source_rows": len(source_rows),
        "filtered_rows": len(combined),
        "total_rows": len(combined),
        "total_events": len(events),
        "shared_events": _shared_events(split_events),
        "splits": split_summary,
    }
    (output / "event_disjoint_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--dev_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--max_target_decision_gap", type=int, default=None)
    args = parser.parse_args()
    summary = build_event_disjoint_package(
        args.package_root,
        args.output_root,
        fold=args.fold,
        num_folds=args.num_folds,
        dev_fraction=args.dev_fraction,
        seed=args.seed,
        max_target_decision_gap=args.max_target_decision_gap,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
