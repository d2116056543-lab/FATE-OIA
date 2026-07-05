from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTION_NAMES = ("maintain_speed", "reduce_speed", "stop_car")


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _dump_pickle(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def _records_from_pickle(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, dict):
        for key in ("samples", "records", "data"):
            if key in obj:
                obj = obj[key]
                break
    return [dict(row) for row in obj]


def _first(row: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    return default


def _action_soft(row: dict[str, Any]) -> list[float]:
    values = _first(
        row,
        ("action_soft_target", "action_soft", "action_distribution", "soft_action"),
        [1.0 / 3.0] * 3,
    )
    values = [float(value) for value in values[:3]]
    total = sum(values)
    if total <= 0:
        return [1.0 / 3.0] * 3
    return [value / total for value in values]


def _action_hard(row: dict[str, Any]) -> int:
    for key in ("action_majority", "action_label", "majority_action"):
        if key in row:
            return int(row[key])
    soft = _action_soft(row)
    return max(range(3), key=lambda idx: soft[idx])


def _action_name(row: dict[str, Any]) -> str:
    name = _first(row, ("action_name",), None)
    if name is not None:
        return str(name)
    hard = _action_hard(row)
    return ACTION_NAMES[hard] if 0 <= hard < len(ACTION_NAMES) else str(hard)


def _decision_keyframe(row: dict[str, Any]) -> int | None:
    value = _first(row, ("decision_keyframe", "decision_frame", "decision_kf"), None)
    return None if value is None else int(value)


def _target_frame(row: dict[str, Any]) -> int | None:
    value = _first(row, ("target_frame",), None)
    return None if value is None else int(value)


def _target_decision_gap(row: dict[str, Any]) -> int | None:
    target = _target_frame(row)
    decision = _decision_keyframe(row)
    if target is None or decision is None:
        return None
    return target - decision


def _explanation_keyframe(row: dict[str, Any]) -> int | None:
    value = _first(row, ("explanation_keyframe", "reasoning_keyframe", "exp_keyframe"), None)
    return None if value is None else int(value)


def _has_explanation_text(row: dict[str, Any]) -> bool:
    return bool(str(_first(row, ("reasoning_text",), "")).strip() or str(_first(row, ("explanation_text",), "")).strip())


def _is_exp29_aligned(row: dict[str, Any], *, alignment_window: int) -> bool:
    target = _target_frame(row)
    explanation = _explanation_keyframe(row)
    if target is None or explanation is None or not _has_explanation_text(row):
        return False
    return abs(explanation - target) <= alignment_window


def _video_id(row: dict[str, Any]) -> str:
    return str(_first(row, ("video_id", "video", "vid"), ""))


def _decision_group(row: dict[str, Any]) -> tuple[str, int]:
    decision = _decision_keyframe(row)
    if decision is None:
        decision = -1
    return _video_id(row), int(decision)


def _target_key(row: dict[str, Any]) -> tuple[str, int]:
    target = _target_frame(row)
    if target is None:
        target = -1
    return _video_id(row), int(target)


def _load_all_records(package_root: Path, splits: tuple[str, ...]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for split in splits:
        path = package_root / "samples" / f"{split}.pkl"
        if not path.exists():
            continue
        for source_index, row in enumerate(_records_from_pickle(_load_pickle(path))):
            copied = dict(row)
            copied.setdefault("source_split", split)
            copied["_protocol_source_split"] = split
            copied["_protocol_source_index"] = source_index
            records.append(copied)
    return records


def _load_source_exp29(package_root: Path, splits: tuple[str, ...]) -> dict[tuple[str, int], dict[str, Any]]:
    source_exp29: dict[tuple[str, int], dict[str, Any]] = {}
    for split in splits:
        path = package_root / "reason_exp29" / f"{split}.pkl"
        if not path.exists():
            continue
        obj = _load_pickle(path)
        if isinstance(obj, dict):
            labels = obj.get("labels", [])
            masks = obj.get("masks", [0.0] * len(labels))
            for idx, label in enumerate(labels):
                mask = float(masks[idx]) if idx < len(masks) else 0.0
                source_exp29[(split, idx)] = {"label": [float(v) for v in label[:29]], "mask": mask}
        elif isinstance(obj, list):
            for idx, item in enumerate(obj):
                if isinstance(item, dict):
                    label = _first(item, ("exp29", "labels", "target", "exp29_target"), [0.0] * 29)
                    mask = _first(item, ("mask", "exp29_mask", "masks"), 0.0)
                    if isinstance(mask, list):
                        mask_value = float(max(mask) if mask else 0.0)
                    else:
                        mask_value = float(mask)
                    source_exp29[(split, idx)] = {"label": [float(v) for v in label[:29]], "mask": mask_value}
    return source_exp29


def _dedupe_by_target(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one row per visual target frame to prevent train/test leakage."""
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for row in sorted(
        records,
        key=lambda item: (
            _video_id(item),
            _target_frame(item) if _target_frame(item) is not None else -1,
            _target_decision_gap(item) if _target_decision_gap(item) is not None else 10**9,
            str(_first(item, ("sample_id", "id"), "")),
        ),
    ):
        best.setdefault(_target_key(row), row)
    return list(best.values())


def _select_test_targets(
    group_rows: list[dict[str, Any]],
    *,
    test_fraction: float,
    min_test_gap: int,
    rng: random.Random,
) -> set[tuple[str, int]]:
    eligible = [row for row in group_rows if (_target_decision_gap(row) or 0) >= min_test_gap]
    if not eligible:
        return set()
    desired = max(1, int(round(len(group_rows) * test_fraction)))
    desired = min(desired, len(eligible))
    # Spread selected frames across the segment instead of taking adjacent rows.
    eligible_sorted = sorted(eligible, key=lambda row: (_target_frame(row) or -1, str(_first(row, ("sample_id", "id"), ""))))
    if desired == len(eligible_sorted):
        chosen = eligible_sorted
    else:
        buckets = []
        for idx in range(desired):
            pos = round(idx * (len(eligible_sorted) - 1) / max(1, desired - 1))
            buckets.append(pos)
        # A deterministic random offset avoids identical class-position bias across groups.
        offset = rng.randrange(len(eligible_sorted)) if len(eligible_sorted) > 1 else 0
        chosen_indices = sorted({(pos + offset) % len(eligible_sorted) for pos in buckets})
        while len(chosen_indices) < desired:
            candidate = rng.randrange(len(eligible_sorted))
            if candidate not in chosen_indices:
                chosen_indices.append(candidate)
        chosen = [eligible_sorted[idx] for idx in sorted(chosen_indices)]
    return {_target_key(row) for row in chosen}


def _unknown_exp_records(count: int) -> list[dict[str, list[float]]]:
    return [{"exp29": [0.0] * 29, "exp29_mask": [0.0] * 29} for _ in range(count)]


def _aligned_exp_record(
    row: dict[str, Any],
    *,
    alignment_window: int,
    source_exp29: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Expose weak Exp supervision only when text and target frame are temporally aligned.

    The reconstructed PSI rows contain forward-filled decisions and backward-filled text.
    Treating every row as a hard explanation label is the failure mode this protocol avoids.
    """
    if not _is_exp29_aligned(row, alignment_window=alignment_window):
        return {"exp29": [0.0] * 29, "exp29_mask": [0.0] * 29}
    source_split = str(_first(row, ("_protocol_source_split", "source_split"), ""))
    source_index = _first(row, ("_protocol_source_index",), None)
    source_item = source_exp29.get((source_split, int(source_index))) if source_index is not None else None
    if not source_item or float(source_item.get("mask", 0.0)) <= 0:
        return {
            "exp29": [0.0] * 29,
            "exp29_mask": [0.0] * 29,
            "exp29_aligned_text": True,
            "exp29_source_mask": 0.0,
            "exp29_alignment_window": alignment_window,
        }
    label = [float(v) for v in source_item["label"][:29]]
    return {
        "exp29": label,
        "exp29_mask": [1.0] * 29,
        "exp29_aligned_text": True,
        "exp29_source_mask": float(source_item.get("mask", 0.0)),
        "exp29_alignment_window": alignment_window,
    }


def _exp_records(
    rows: list[dict[str, Any]],
    *,
    exp29_policy: str,
    alignment_window: int,
    source_exp29: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    if exp29_policy == "all_unknown":
        return _unknown_exp_records(len(rows))
    if exp29_policy == "aligned_only":
        return [
            _aligned_exp_record(row, alignment_window=alignment_window, source_exp29=source_exp29)
            for row in rows
        ]
    raise ValueError(f"Unsupported exp29_policy: {exp29_policy}")


def _stamp_record(
    row: dict[str, Any],
    *,
    protocol_name: str,
    split: str,
    exp29_policy: str,
    exp29_alignment_window: int,
    duplicate_index: int = 0,
) -> dict[str, Any]:
    out = copy.deepcopy(row)
    hard = _action_hard(out)
    out["action_majority"] = hard
    out["action_name"] = ACTION_NAMES[hard]
    out["action_soft_target"] = _action_soft(out)
    out["target_decision_gap"] = int(_target_decision_gap(out) or 0)
    out["protocol_name"] = protocol_name
    out["protocol_split"] = split
    out["exp29_policy"] = exp29_policy
    out["exp29_alignment_window"] = exp29_alignment_window
    out["exp29_temporally_aligned"] = _is_exp29_aligned(out, alignment_window=exp29_alignment_window)
    out["exp29_supervised"] = False
    if duplicate_index:
        out["protocol_duplicate_index"] = duplicate_index
        out["sample_id"] = f"{_first(out, ('sample_id', 'id'), 'row')}_dup{duplicate_index}"
    return out


def _oversample_stop_train(
    train_rows: list[dict[str, Any]],
    *,
    target_stop_rate: float,
    protocol_name: str,
    exp29_policy: str,
    exp29_alignment_window: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if target_stop_rate <= 0:
        return train_rows
    stop_rows = [row for row in train_rows if _action_hard(row) == 2]
    if not stop_rows:
        return train_rows
    total = len(train_rows)
    stop_count = len(stop_rows)
    if stop_count / max(1, total) >= target_stop_rate:
        return train_rows
    needed = math.ceil((target_stop_rate * total - stop_count) / max(1e-9, 1.0 - target_stop_rate))
    augmented = list(train_rows)
    for idx in range(needed):
        source = copy.deepcopy(stop_rows[idx % len(stop_rows)])
        if idx and idx % len(stop_rows) == 0:
            rng.shuffle(stop_rows)
        augmented.append(
            _stamp_record(
                source,
                protocol_name=protocol_name,
                split="train",
                exp29_policy=exp29_policy,
                exp29_alignment_window=exp29_alignment_window,
                duplicate_index=idx + 1,
            )
        )
    return augmented


def build_segment_frame_protocol(
    *,
    source_package_root: Path,
    output_root: Path,
    protocol_name: str = "segment_frame_split_gap30_stop20",
    max_gap: int = 30,
    test_fraction: float = 0.20,
    min_test_gap: int = 3,
    target_stop_train_rate: float = 0.20,
    seed: int = 1,
    source_splits: tuple[str, ...] = ("train", "val", "test"),
    exp29_policy: str = "all_unknown",
    exp29_alignment_window: int = 3,
) -> dict[str, Any]:
    rng = random.Random(seed)
    all_records = _load_all_records(source_package_root, source_splits)
    source_exp29 = _load_source_exp29(source_package_root, source_splits)
    filtered = [
        row
        for row in all_records
        if (gap := _target_decision_gap(row)) is not None and 0 <= gap <= max_gap
    ]
    deduped = _dedupe_by_target(filtered)

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in deduped:
        grouped[_decision_group(row)].append(row)

    train_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        rows = sorted(grouped[group_key], key=lambda row: (_target_frame(row) or -1, str(_first(row, ("sample_id", "id"), ""))))
        test_keys = _select_test_targets(rows, test_fraction=test_fraction, min_test_gap=min_test_gap, rng=rng)
        for row in rows:
            split = "test" if _target_key(row) in test_keys else "train"
            stamped = _stamp_record(
                row,
                protocol_name=protocol_name,
                split=split,
                exp29_policy=exp29_policy,
                exp29_alignment_window=exp29_alignment_window,
            )
            if split == "test":
                test_rows.append(stamped)
            else:
                train_rows.append(stamped)

    train_rows = _oversample_stop_train(
        train_rows,
        target_stop_rate=target_stop_train_rate,
        protocol_name=protocol_name,
        exp29_policy=exp29_policy,
        exp29_alignment_window=exp29_alignment_window,
        rng=rng,
    )

    splits = {"train": train_rows, "val": [], "test": test_rows}
    split_exp_records: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        exp_records = _exp_records(
            rows,
            exp29_policy=exp29_policy,
            alignment_window=exp29_alignment_window,
            source_exp29=source_exp29,
        )
        split_exp_records[split] = exp_records
        for row, exp in zip(rows, exp_records):
            row["exp29_supervised"] = bool(sum(exp.get("exp29_mask", [])) > 0)
        _dump_pickle(output_root / protocol_name / "samples" / f"{split}.pkl", rows)
        _dump_pickle(output_root / protocol_name / "reason_exp29" / f"{split}.pkl", exp_records)

    summary = _summarize_protocol(
        source_package_root=source_package_root,
        output_package_root=output_root / protocol_name,
        protocol_name=protocol_name,
        max_gap=max_gap,
        min_test_gap=min_test_gap,
        test_fraction=test_fraction,
        target_stop_train_rate=target_stop_train_rate,
        seed=seed,
        source_rows=len(all_records),
        filtered_rows=len(filtered),
        deduped_rows=len(deduped),
        splits=splits,
        exp29_policy=exp29_policy,
        exp29_alignment_window=exp29_alignment_window,
    )
    (output_root / protocol_name / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _summarize_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(_action_hard(row) for row in rows)
    gaps = [_target_decision_gap(row) for row in rows if _target_decision_gap(row) is not None]
    return {
        "rows": len(rows),
        "videos": len({_video_id(row) for row in rows}),
        "decision_groups": len({_decision_group(row) for row in rows}),
        "unique_target_frames": len({_target_key(row) for row in rows}),
        "action_counts": {ACTION_NAMES[idx]: int(counts.get(idx, 0)) for idx in range(3)},
        "action_rates": {
            ACTION_NAMES[idx]: (float(counts.get(idx, 0)) / len(rows) if rows else 0.0)
            for idx in range(3)
        },
        "gap_min": min(gaps) if gaps else None,
        "gap_max": max(gaps) if gaps else None,
        "exp29_aligned_rows": sum(bool(row.get("exp29_temporally_aligned")) for row in rows),
        "exp29_supervised_rows": sum(bool(row.get("exp29_supervised")) for row in rows),
        "nonempty_text_rows": sum(_has_explanation_text(row) for row in rows),
    }


def _summarize_protocol(
    *,
    source_package_root: Path,
    output_package_root: Path,
    protocol_name: str,
    max_gap: int,
    min_test_gap: int,
    test_fraction: float,
    target_stop_train_rate: float,
    seed: int,
    source_rows: int,
    filtered_rows: int,
    deduped_rows: int,
    splits: dict[str, list[dict[str, Any]]],
    exp29_policy: str,
    exp29_alignment_window: int,
) -> dict[str, Any]:
    train_targets = {_target_key(row) for row in splits["train"]}
    test_targets = {_target_key(row) for row in splits["test"]}
    train_groups = {_decision_group(row) for row in splits["train"]}
    test_groups = {_decision_group(row) for row in splits["test"]}
    return {
        "protocol_name": protocol_name,
        "source_package_root": str(source_package_root),
        "output_package_root": str(output_package_root),
        "seed": seed,
        "max_gap": max_gap,
        "min_test_gap": min_test_gap,
        "test_fraction": test_fraction,
        "target_stop_train_rate": target_stop_train_rate,
        "source_rows": source_rows,
        "filtered_rows": filtered_rows,
        "deduped_rows": deduped_rows,
        "splits": {split: _summarize_split(rows) for split, rows in splits.items()},
        "leakage": {
            "target_frame_overlap_train_test": len(train_targets & test_targets),
            "decision_group_overlap_train_test": len(train_groups & test_groups),
        },
        "exp29_policy": exp29_policy,
        "exp29_alignment_window": exp29_alignment_window,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reproducible PSI segment-frame action protocol package.")
    parser.add_argument("--source_package_root", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--protocol_name", default="segment_frame_split_gap30_stop20")
    parser.add_argument("--max_gap", default=30, type=int)
    parser.add_argument("--test_fraction", default=0.20, type=float)
    parser.add_argument("--min_test_gap", default=3, type=int)
    parser.add_argument("--target_stop_train_rate", default=0.20, type=float)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--exp29_policy", choices=("all_unknown", "aligned_only"), default="all_unknown")
    parser.add_argument("--exp29_alignment_window", default=3, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_segment_frame_protocol(
        source_package_root=args.source_package_root,
        output_root=args.output_root,
        protocol_name=args.protocol_name,
        max_gap=args.max_gap,
        test_fraction=args.test_fraction,
        min_test_gap=args.min_test_gap,
        target_stop_train_rate=args.target_stop_train_rate,
        seed=args.seed,
        exp29_policy=args.exp29_policy,
        exp29_alignment_window=args.exp29_alignment_window,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
