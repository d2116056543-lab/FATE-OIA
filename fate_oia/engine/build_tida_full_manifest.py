from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from typing import Callable, Sequence

from fate_oia.datasets.tida_clip_manifest import (
    TIDAClipRecord,
    normalize_source_id,
    partition_train_records,
    validate_records,
    write_manifest,
)


Probe = Callable[[Path], tuple[float, int]]


def _clip_key(path: Path) -> str:
    return path.stem.removesuffix("_prev5s")


def _resolve_label(clip: Path, label_dir: Path) -> Path | None:
    for stem in (clip.stem, _clip_key(clip)):
        candidate = label_dir / f"{stem}.json"
        if candidate.is_file():
            return candidate
    return None


def _resolve_target(clip: Path, frame_dir: Path, label: dict) -> Path | None:
    candidates = []
    declared = label.get("last_frame_image_path")
    if declared:
        candidates.append(Path(str(declared)))
    for stem in (clip.stem, _clip_key(clip)):
        candidates.extend(frame_dir / f"{stem}{suffix}" for suffix in (".jpg", "_lastframe.jpg", ".png"))
    return next((path for path in candidates if path.is_file()), None)


def _source_id(label: dict, key: str) -> str:
    declared = label.get("video_stem") or label.get("video_path")
    if declared:
        return normalize_source_id(str(declared))
    image_stem = Path(str(label.get("image_name") or key)).stem
    return normalize_source_id(re.sub(r"_[0-9]+$", "", image_stem))


def probe_video(path: Path) -> tuple[float, int]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("video_open_failed")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()
    if fps <= 0 or frames < 2:
        raise ValueError("invalid_video_timing")
    return fps, frames


def discover_labeled_clips(
    root: str | Path, *, probe: Probe = probe_video, probe_workers: int = 1,
    progress_every: int = 0,
) -> tuple[list[TIDAClipRecord], dict]:
    root = Path(root)
    records: list[TIDAClipRecord] = []
    candidates = []
    rejections: Counter[str] = Counter()
    clip_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    for split in ("train", "test"):
        clip_dir = root / "clips_prev5s" / split
        label_dir = root / "labels" / split
        frame_dir = root / "last_frames" / split
        labels = list(label_dir.glob("*.json"))
        label_counts[split] = len(labels)
        for clip in sorted(clip_dir.glob("*.mp4")):
            clip_counts[split] += 1
            label_path = _resolve_label(clip, label_dir)
            if label_path is None:
                rejections["missing_label"] += 1
                continue
            try:
                label = json.loads(label_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                rejections["invalid_label_json"] += 1
                continue
            action, reason = label.get("action_4"), label.get("reason_21")
            if not isinstance(action, (list, tuple)) or len(action) != 4 or not isinstance(reason, (list, tuple)) or len(reason) != 21:
                rejections["null_or_invalid_target"] += 1
                continue
            target = _resolve_target(clip, frame_dir, label)
            if target is None:
                rejections["missing_last_frame"] += 1
                continue
            key = _clip_key(clip)
            source_id = _source_id(label, key)
            if not source_id:
                rejections["missing_source_id"] += 1
                continue
            candidates.append((split, clip, label_path, label, target, key, source_id, action, reason))

    def build_record(candidate):
        split, clip, label_path, label, target, key, source_id, action, reason = candidate
        try:
            fps, num_frames = probe(clip)
        except (OSError, ValueError):
            return None, candidate
        return TIDAClipRecord(
                official_split=split,
                partition="unassigned",
                file_name=Path(str(label.get("image_name") or f"{key}.jpg")).name,
                target_image_path=target,
                clip_path=clip,
                source_video_id=source_id,
                duration_seconds=float(num_frames / fps),
                fps=float(fps),
                num_frames=int(num_frames),
                target_timestamp_seconds=float((num_frames - 1) / fps),
                target_frame_index=int(num_frames - 1),
                action=tuple(float(value) for value in action),
                reason=tuple(float(value) for value in reason),
                source_batch=root.name,
                source_manifest_path=str(label_path),
            ), None

    failed_probes = []
    if int(probe_workers) > 1:
        with ThreadPoolExecutor(max_workers=int(probe_workers)) as executor:
            iterator = executor.map(build_record, candidates)
            for index, (record, failed) in enumerate(iterator, start=1):
                if record is None:
                    failed_probes.append(failed)
                else:
                    records.append(record)
                if progress_every and index % int(progress_every) == 0:
                    print(f"manifest_probe={index}/{len(candidates)}", flush=True)
    else:
        for index, candidate in enumerate(candidates, start=1):
            record, failed = build_record(candidate)
            if record is None:
                failed_probes.append(failed)
            else:
                records.append(record)
            if progress_every and index % int(progress_every) == 0:
                print(f"manifest_probe={index}/{len(candidates)}", flush=True)
    retried_video_probes = len(failed_probes)
    rejected_video_paths = []
    for candidate in failed_probes:
        record, _ = build_record(candidate)
        if record is None:
            rejections["invalid_video"] += 1
            rejected_video_paths.append(str(candidate[1]))
        else:
            records.append(record)
    accepted = Counter(record.official_split for record in records)
    return records, {
        "clip_files_by_split": dict(clip_counts),
        "label_files_by_split": dict(label_counts),
        "accepted_by_split": {split: int(accepted[split]) for split in ("train", "test")},
        "rejections": dict(rejections),
        "retried_video_probes": retried_video_probes,
        "rejected_video_paths": rejected_video_paths,
    }


def partition_official_source_disjoint(
    records: Sequence[TIDAClipRecord], *, calib_count: int, audit_count: int, seed: int
) -> tuple[list[TIDAClipRecord], dict]:
    test_records = [record for record in records if record.official_split == "test"]
    test_sources = {normalize_source_id(record.source_video_id) for record in test_records}
    train_candidates = [record for record in records if record.official_split == "train"]
    clean_train = [
        record for record in train_candidates
        if normalize_source_id(record.source_video_id) not in test_sources
    ]
    excluded = len(train_candidates) - len(clean_train)
    partitioned_train = partition_train_records(
        clean_train, seed=seed, calib_count=calib_count, audit_count=audit_count
    )
    partitioned_test = [
        TIDAClipRecord.from_dict({**record.to_dict(), "partition": "test"})
        for record in test_records
    ]
    output = sorted(partitioned_train + partitioned_test, key=lambda row: (row.partition, row.file_name))
    counts = Counter(record.partition for record in output)
    return output, {
        "excluded_train_test_source_overlap": excluded,
        "partition_counts": dict(counts),
        "train_source_count": len({record.source_video_id for record in partitioned_train}),
        "test_source_count": len(test_sources),
    }


def partition_official_split(
    records: Sequence[TIDAClipRecord], *, calib_count: int, audit_count: int, seed: int
) -> tuple[list[TIDAClipRecord], dict]:
    train_records = [record for record in records if record.official_split == "train"]
    test_records = [record for record in records if record.official_split == "test"]
    partitioned_train = partition_train_records(
        train_records, seed=seed, calib_count=calib_count, audit_count=audit_count
    )
    partitioned_test = [
        TIDAClipRecord.from_dict({**record.to_dict(), "partition": "test"})
        for record in test_records
    ]
    train_sources = {normalize_source_id(record.source_video_id) for record in train_records}
    test_sources = {normalize_source_id(record.source_video_id) for record in test_records}
    overlap = train_sources & test_sources
    output = sorted(partitioned_train + partitioned_test, key=lambda row: (row.partition, row.file_name))
    counts = Counter(record.partition for record in output)
    return output, {
        "excluded_train_test_source_overlap": 0,
        "diagnostic_cross_split_source_count": len(overlap),
        "diagnostic_cross_split_train_rows": sum(
            normalize_source_id(record.source_video_id) in overlap for record in train_records
        ),
        "diagnostic_cross_split_test_rows": sum(
            normalize_source_id(record.source_video_id) in overlap for record in test_records
        ),
        "partition_counts": dict(counts),
        "official_train_total": len(train_records),
        "official_test_total": len(test_records),
    }


def validate_official_split_records(records: Sequence[TIDAClipRecord]) -> dict:
    keys = set()
    errors = []
    for record in records:
        key = (record.official_split, record.file_name.lower())
        if key in keys:
            errors.append(f"duplicate key: {key}")
        keys.add(key)
        if not record.target_image_path.is_file() or not record.clip_path.is_file():
            errors.append(f"missing path: {record.file_name}")
        if len(record.action) != 4 or len(record.reason) != 21:
            errors.append(f"invalid target dimensions: {record.file_name}")
    return {"pass": not errors, "count": len(records), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calib-count", type=int, default=779)
    parser.add_argument("--audit-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--probe-workers", type=int, default=8)
    parser.add_argument(
        "--split-policy", choices=("official", "source_disjoint"), default="official"
    )
    args = parser.parse_args()

    records, discovery = discover_labeled_clips(
        args.dataset_root, probe_workers=args.probe_workers, progress_every=500
    )
    partitioner = (
        partition_official_split
        if args.split_policy == "official"
        else partition_official_source_disjoint
    )
    partitioned, partition_audit = partitioner(
        records, calib_count=args.calib_count, audit_count=args.audit_count, seed=args.seed
    )
    validation = (
        validate_official_split_records(partitioned)
        if args.split_policy == "official"
        else validate_records(partitioned, require_files=True)
    )
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "tida_full_primary_manifest.jsonl"
    audit_path = output_dir / "tida_full_manifest_audit.json"
    write_manifest(manifest_path, partitioned)
    audit = {
        "pass": bool(validation["pass"]),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "split_policy": args.split_policy,
        "discovery": discovery,
        "partition": partition_audit,
        "validation": validation,
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=True))
    if not audit["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
