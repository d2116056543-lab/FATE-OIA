from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from fate_oia.datasets.tida_clip_manifest import (
    TIDAClipRecord,
    load_manifest,
    normalize_source_id,
    validate_records,
    write_manifest,
)
from fate_oia.engine.build_tida_balanced_pilot_manifest import merge_source_matched_records


def build_source_complete_records(
    primary_records: list[TIDAClipRecord],
    legacy_records: list[TIDAClipRecord],
) -> tuple[list[TIDAClipRecord], dict[str, object]]:
    """Add only source-safe Batch1/2 train rows while preserving the fixed test set."""
    merged, merge_audit = merge_source_matched_records(
        primary_records, legacy_records, allow_existing_train_source=True
    )
    train_sources = {
        normalize_source_id(row.source_video_id) for row in merged if row.partition != "test"
    }
    test_sources = {
        normalize_source_id(row.source_video_id) for row in merged if row.partition == "test"
    }
    overlap = sorted(train_sources & test_sources)
    if overlap:
        raise ValueError(f"source-complete manifest leaked {len(overlap)} test sources")
    file_names = [row.file_name.lower() for row in merged]
    duplicate_files = sorted(name for name, count in Counter(file_names).items() if count > 1)
    if duplicate_files:
        raise ValueError(f"source-complete manifest has duplicate files: {duplicate_files[:5]}")
    validation = validate_records(merged, require_files=False)
    audit = {
        "pass": bool(validation["pass"] and not overlap and not duplicate_files),
        "total_count": len(merged),
        "partition_counts": {
            partition: sum(row.partition == partition for row in merged)
            for partition in ("train_core", "train_calib", "train_audit", "test")
        },
        "source_counts": {
            "train": len(train_sources),
            "test": len(test_sources),
        },
        "train_test_source_overlap": len(overlap),
        "duplicate_file_count": len(duplicate_files),
        "validation_errors": validation["errors"],
        "source_matched_merge": merge_audit,
    }
    if not audit["pass"]:
        raise RuntimeError(f"source-complete manifest failed validation: {audit}")
    return merged, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-input", required=True)
    parser.add_argument("--legacy-input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    args = parser.parse_args()
    records, audit = build_source_complete_records(
        load_manifest(args.primary_input), load_manifest(args.legacy_input)
    )
    write_manifest(args.output, records)
    Path(args.audit_output).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
