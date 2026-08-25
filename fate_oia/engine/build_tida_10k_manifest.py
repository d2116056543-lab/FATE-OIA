from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from fate_oia.datasets.tida_clip_manifest import (
    TIDAClipRecord,
    _exact_group_subset,
    _group_order,
    _phash64,
    file_sha256,
    load_manifest,
    normalize_source_id,
    write_manifest,
)
from fate_oia.datasets.tida_official_labels import OfficialLabelMap, load_official_label_map


PARTITION_KEYS = (
    "train_core", "train_calib", "train_audit", "expanded_test", "legacy_overlap_excluded"
)


def assign_dual_evaluation_groups(
    rows: Sequence[dict[str, Any]], *, legacy_test_source_ids: set[str],
    expanded_test_count: int, calib_count: int, audit_count: int, seed: int,
) -> dict[str, list[dict[str, Any]]]:
    normalized_legacy = {normalize_source_id(value) for value in legacy_test_source_ids}
    result = {key: [] for key in PARTITION_KEYS}
    groups: dict[str, list[dict[str, Any]]] = {}
    for source in rows:
        row = dict(source)
        source_id = normalize_source_id(str(row["source_video_id"]))
        row["source_video_id"] = source_id
        if source_id in normalized_legacy:
            result["legacy_overlap_excluded"].append(row)
        else:
            groups.setdefault(source_id, []).append(row)
    ordered = _group_order(groups, seed)
    calib_indices = set(_exact_group_subset(ordered, int(calib_count))) if calib_count else set()
    after_calib = [entry for index, entry in enumerate(ordered) if index not in calib_indices]
    audit_indices = set(_exact_group_subset(after_calib, int(audit_count))) if audit_count else set()
    after_audit = [entry for index, entry in enumerate(after_calib) if index not in audit_indices]
    expanded_indices = (
        set(_exact_group_subset(after_audit, int(expanded_test_count))) if expanded_test_count else set()
    )
    for index, (_, members) in enumerate(ordered):
        if index in calib_indices:
            result["train_calib"].extend(members)
    for index, (_, members) in enumerate(after_calib):
        if index in audit_indices:
            result["train_audit"].extend(members)
    for index, (_, members) in enumerate(after_audit):
        result["expanded_test" if index in expanded_indices else "train_core"].extend(members)
    for key in result:
        result[key] = sorted(result[key], key=lambda row: (row["source_video_id"], row["file_name"]))
    return result


def attach_labels(row: dict[str, Any], labels: OfficialLabelMap) -> dict[str, Any]:
    name = Path(str(row["image_name"])).name.lower()
    if name not in labels:
        raise ValueError(f"official labels missing for {name}")
    action, reason = labels[name]
    output = dict(row)
    output["file_name"] = Path(str(row["image_name"])).name
    output["action_4"] = list(action)
    output["reason_21"] = list(reason)
    output["source_video_id"] = normalize_source_id(
        str(row.get("source_video_id") or row.get("stem") or row.get("video_path"))
    )
    return output


def apply_repair_overlay(
    rows: Sequence[dict[str, Any]], repairs: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    result = []
    for source in rows:
        row = dict(source)
        repair = repairs.get(Path(str(row["image_name"])).name.lower())
        if repair is not None:
            row["clip_path"] = str(repair["clip_path"])
            row["clip"] = {**dict(row.get("clip", {})), "ok": True, "frames_written": int(repair["frames_written"])}
            row["match"] = {**dict(row.get("match", {})), "fps": float(repair["fps"])}
            row["repair_applied"] = True
        result.append(row)
    return result


def filter_source_quality(
    rows: Sequence[dict[str, Any]], *, max_mse: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        reasons: list[str] = []
        frames_written = int(row.get("clip", {}).get("frames_written", 0))
        endpoint_mse = float(row.get("match", {}).get("mse", float("inf")))
        if frames_written < 2:
            reasons.append("zero_frame_clip")
        if endpoint_mse > float(max_mse):
            reasons.append("endpoint_mse")
        if reasons:
            row["quality_rejection_reasons"] = reasons
            rejected.append(row)
        else:
            accepted.append(row)
    return accepted, rejected


def source_row_to_record(
    row: dict[str, Any], partition: str, *, compute_hashes: bool = True
) -> TIDAClipRecord:
    import numpy as np
    from PIL import Image

    clip_path = Path(str(row["clip_path"]))
    target_path = Path(str(row.get("last_frame_image_path") or row.get("image_path")))
    fps = float(row.get("match", {}).get("fps", 0.0))
    num_frames = int(row.get("clip", {}).get("frames_written", 0))
    if fps <= 0 or num_frames < 2:
        raise ValueError(f"invalid clip timing for {row['file_name']}: fps={fps}, frames={num_frames}")
    clip_hash = file_sha256(clip_path) if compute_hashes else ""
    endpoint_phash = (
        _phash64(np.asarray(Image.open(target_path).convert("RGB"))) if compute_hashes else 0
    )
    source_manifest = str(row.get("source_manifest_path", ""))
    return TIDAClipRecord(
        official_split=str(row["split"]).lower(),
        partition=partition,
        file_name=str(row["file_name"]),
        target_image_path=target_path,
        clip_path=clip_path,
        source_video_id=normalize_source_id(str(row["source_video_id"])),
        duration_seconds=float(num_frames / fps),
        fps=fps,
        num_frames=num_frames,
        target_timestamp_seconds=float((num_frames - 1) / fps),
        target_frame_index=num_frames - 1,
        action=tuple(float(value) for value in row["action_4"]),
        reason=tuple(float(value) for value in row["reason_21"]),
        clip_sha256=clip_hash,
        endpoint_phash=endpoint_phash,
        source_batch=Path(source_manifest).parent.name if source_manifest else "",
        source_manifest_path=source_manifest,
        source_row_index=int(row.get("source_row_index", -1)),
    )


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", action="append", required=True)
    parser.add_argument("--legacy-manifest", required=True)
    parser.add_argument("--oia-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--repair-manifest")
    parser.add_argument("--expanded-test-count", type=int, default=1328)
    parser.add_argument("--calib-count", type=int, default=779)
    parser.add_argument("--audit-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--max-source-mse", type=float, default=0.0021)
    parser.add_argument("--compute-hashes", action="store_true")
    args = parser.parse_args()
    oia_root = Path(args.oia_root)
    labels = {
        split: load_official_label_map(
            oia_root / f"{split}_25k_images_actions.json",
            oia_root / f"{split}_25k_images_reasons.json",
        )
        for split in ("train", "test")
    }
    rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for manifest_value in args.source_manifest:
        manifest = Path(manifest_value)
        for row_index, row in enumerate(_rows(manifest)):
            split = str(row["split"]).lower()
            repaired = attach_labels(row, labels[split])
            repaired["source_manifest_path"] = str(manifest.resolve())
            repaired["source_row_index"] = row_index
            key = (split, str(row["image_name"]).lower())
            if key in seen_keys:
                raise ValueError(f"duplicate split/image key: {key}")
            seen_keys.add(key)
            rows.append(repaired)
    source_rows_total = len(rows)
    repair_map: dict[str, dict[str, Any]] = {}
    if args.repair_manifest:
        for repair in _rows(Path(args.repair_manifest)):
            name = Path(str(repair["image_name"])).name.lower()
            if name in repair_map:
                raise ValueError(f"duplicate repair overlay: {name}")
            repair_map[name] = repair
        rows = apply_repair_overlay(rows, repair_map)
    rows, quality_rejected = filter_source_quality(rows, max_mse=args.max_source_mse)
    legacy_records = load_manifest(Path(args.legacy_manifest))
    legacy_test_records = [record for record in legacy_records if record.partition == "test"]
    legacy_test = [record.to_dict() for record in legacy_test_records]
    if len(legacy_test) != 885:
        raise ValueError(f"legacy test must contain exactly 885 rows, got {len(legacy_test)}")
    partitions = assign_dual_evaluation_groups(
        rows,
        legacy_test_source_ids={str(row["source_video_id"]) for row in legacy_test},
        expanded_test_count=args.expanded_test_count,
        calib_count=args.calib_count,
        audit_count=args.audit_count,
        seed=args.seed,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "quality_rejected.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in quality_rejected),
        encoding="utf-8",
    )
    for name, values in partitions.items():
        (output_dir / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8"
        )
    primary_records = list(legacy_test_records)
    for partition in ("train_core", "train_calib", "train_audit"):
        primary_records.extend(
            source_row_to_record(row, partition, compute_hashes=args.compute_hashes)
            for row in partitions[partition]
        )
    primary_records.sort(key=lambda record: (record.partition, record.file_name.lower()))
    expanded_records = [
        source_row_to_record(row, "test", compute_hashes=args.compute_hashes)
        for row in partitions["expanded_test"]
    ]
    expanded_records.sort(key=lambda record: record.file_name.lower())
    primary_manifest = output_dir / "tida_10k_primary_manifest.jsonl"
    expanded_manifest = output_dir / "tida_10k_expanded_test_manifest.jsonl"
    write_manifest(primary_manifest, primary_records)
    write_manifest(expanded_manifest, expanded_records)
    train_source_ids = {
        record.source_video_id for record in primary_records if record.partition != "test"
    }
    legacy_source_ids = {record.source_video_id for record in legacy_test_records}
    expanded_source_ids = {record.source_video_id for record in expanded_records}
    audit = {
        "pass": True,
        "source_rows": source_rows_total,
        "accepted_source_rows": len(rows),
        "quality_rejected_rows": len(quality_rejected),
        "quality_rejection_counts": {
            reason: sum(
                reason in row["quality_rejection_reasons"] for row in quality_rejected
            )
            for reason in ("endpoint_mse", "zero_frame_clip")
        },
        "max_source_mse": float(args.max_source_mse),
        "content_hashes_computed": bool(args.compute_hashes),
        "legacy_test_rows": len(legacy_test),
        "partition_counts": {key: len(value) for key, value in partitions.items()},
        "null_action": sum(row.get("action_4") is None for row in rows),
        "null_reason": sum(row.get("reason_21") is None for row in rows),
        "missing_clip": sum(not Path(row["clip_path"]).is_file() for row in rows),
        "missing_target": sum(
            not Path(row.get("last_frame_image_path") or row.get("image_path", "")).is_file() for row in rows
        ),
        "duplicate_split_image_keys": source_rows_total - len(seen_keys),
        "primary_manifest_rows": len(primary_records),
        "expanded_manifest_rows": len(expanded_records),
        "repair_overlay_rows": len(repair_map),
        "repair_applied_rows": sum(bool(row.get("repair_applied")) for row in rows),
        "train_legacy_source_overlap": len(train_source_ids & legacy_source_ids),
        "train_expanded_source_overlap": len(train_source_ids & expanded_source_ids),
        "legacy_expanded_source_overlap": len(legacy_source_ids & expanded_source_ids),
        "action_positive_counts": [
            int(sum(float(row["action_4"][index]) for row in rows)) for index in range(4)
        ],
        "reason_positive_counts": [
            int(sum(float(row["reason_21"][index]) for row in rows)) for index in range(21)
        ],
        "primary_manifest": str(primary_manifest.resolve()),
        "expanded_test_manifest": str(expanded_manifest.resolve()),
    }
    audit["pass"] = all(audit[key] == 0 for key in (
        "null_action", "null_reason", "missing_clip", "missing_target", "duplicate_split_image_keys",
        "train_legacy_source_overlap", "train_expanded_source_overlap", "legacy_expanded_source_overlap",
    ))
    (output_dir / "inventory_audit_10k.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
