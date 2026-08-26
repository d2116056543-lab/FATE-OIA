from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

from ..datasets.tida_clip_manifest import (
    TIDAClipRecord,
    load_manifest,
    normalize_source_id,
    validate_records,
    write_manifest,
)


def source_domain(record: TIDAClipRecord) -> str:
    text = f"{record.source_batch} {record.clip_path}".lower()
    if "bdd_oia_1000_train_test" in text:
        return "batch1"
    for domain in ("batch2", "batch3", "batch4", "batch5"):
        if domain in text:
            return domain
    return "other"


def merge_source_matched_records(
    expanded_records: list[TIDAClipRecord],
    legacy_records: list[TIDAClipRecord],
    *,
    allow_existing_train_source: bool = False,
) -> tuple[list[TIDAClipRecord], dict[str, object]]:
    """Add source-disjoint Batch1/2 training rows to the expanded protocol."""
    test_sources = {
        normalize_source_id(row.source_video_id)
        for row in expanded_records if row.partition == "test"
    }
    existing_train_sources = {
        normalize_source_id(row.source_video_id)
        for row in expanded_records if row.partition != "test"
    }
    existing_train_partitions = {
        normalize_source_id(row.source_video_id): row.partition
        for row in expanded_records if row.partition != "test"
    }
    merged = list(expanded_records)
    seen_files = {row.file_name.lower() for row in merged}
    added_counts: dict[str, int] = defaultdict(int)
    excluded_test_source_overlap = 0
    excluded_existing_source_overlap = 0
    excluded_nonlegacy_domain = 0
    excluded_duplicate_file = 0
    reassigned_existing_source_partition = 0
    for row in legacy_records:
        domain = source_domain(row)
        source_id = normalize_source_id(row.source_video_id)
        if row.partition == "test":
            continue
        if domain not in {"batch1", "batch2"}:
            excluded_nonlegacy_domain += 1
            continue
        if source_id in test_sources:
            excluded_test_source_overlap += 1
            continue
        if not allow_existing_train_source and source_id in existing_train_sources:
            excluded_existing_source_overlap += 1
            continue
        if row.file_name.lower() in seen_files:
            excluded_duplicate_file += 1
            continue
        if allow_existing_train_source and source_id in existing_train_partitions:
            owner_partition = existing_train_partitions[source_id]
            if row.partition != owner_partition:
                row = replace(row, partition=owner_partition)
                reassigned_existing_source_partition += 1
        merged.append(row)
        seen_files.add(row.file_name.lower())
        existing_train_sources.add(source_id)
        existing_train_partitions[source_id] = row.partition
        added_counts[domain] += 1
    merged.sort(key=lambda row: (row.partition, row.file_name.lower()))
    train_sources = {
        normalize_source_id(row.source_video_id)
        for row in merged if row.partition != "test"
    }
    overlap = train_sources & test_sources
    if overlap:
        raise ValueError(f"source-matched merge leaked {len(overlap)} test sources")
    return merged, {
        "added_source_counts": dict(sorted(added_counts.items())),
        "excluded_test_source_overlap": excluded_test_source_overlap,
        "excluded_existing_source_overlap": excluded_existing_source_overlap,
        "excluded_nonlegacy_domain": excluded_nonlegacy_domain,
        "excluded_duplicate_file": excluded_duplicate_file,
        "reassigned_existing_source_partition": reassigned_existing_source_partition,
        "train_test_source_overlap": 0,
    }


def _stable_order(rows: list[TIDAClipRecord], seed: int, namespace: str) -> list[TIDAClipRecord]:
    return sorted(
        rows,
        key=lambda row: (
            sha256(f"{seed}:{namespace}:{row.file_name.lower()}".encode()).hexdigest(),
            row.file_name.lower(),
        ),
    )


def _action_balanced_subset(
    rows: list[TIDAClipRecord], count: int, seed: int, domain: str
) -> list[TIDAClipRecord]:
    buckets: dict[tuple[float, ...], list[TIDAClipRecord]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row.action)].append(row)
    ordered_buckets = [
        _stable_order(values, seed, f"{domain}:{action}")
        for action, values in sorted(buckets.items())
    ]
    selected: list[TIDAClipRecord] = []
    cursor = 0
    while len(selected) < count and any(ordered_buckets):
        bucket = ordered_buckets[cursor % len(ordered_buckets)]
        if bucket:
            selected.append(bucket.pop())
        cursor += 1
    if len(selected) != count:
        raise ValueError(f"domain {domain} provides only {len(selected)} of {count} rows")
    return selected


def select_balanced_train_records(
    records: list[TIDAClipRecord], *, train_count: int, seed: int
) -> tuple[list[TIDAClipRecord], dict[str, object]]:
    groups: dict[str, list[TIDAClipRecord]] = defaultdict(list)
    for record in records:
        if record.partition != "train_core":
            raise ValueError("balanced pilot selection accepts train_core rows only")
        domain = source_domain(record)
        if domain != "other":
            groups[domain].append(record)
    domains = sorted(groups)
    if not domains:
        raise ValueError("no recognized video train domains were found")
    if sum(len(groups[domain]) for domain in domains) < int(train_count):
        raise ValueError("recognized video train domains cannot satisfy train_count")
    quotas = {domain: 0 for domain in domains}
    while sum(quotas.values()) < int(train_count):
        available = [
            domain for domain in domains if quotas[domain] < len(groups[domain])
        ]
        if not available:
            raise ValueError("capacity-aware source allocation exhausted unexpectedly")
        domain = min(available, key=lambda name: (quotas[name], name))
        quotas[domain] += 1
    selected = []
    for domain in domains:
        selected.extend(_action_balanced_subset(groups[domain], quotas[domain], seed, domain))
    selected = sorted(selected, key=lambda row: row.file_name.lower())
    return selected, {
        "train_count": len(selected),
        "seed": int(seed),
        "source_capacities": {domain: len(groups[domain]) for domain in domains},
        "source_quotas": quotas,
        "source_counts": {
            domain: sum(source_domain(row) == domain for row in selected)
            for domain in domains
        },
        "action_set_counts": {
            str(tuple(int(value) for value in action)): sum(row.action == action for row in selected)
            for action in sorted({row.action for row in selected})
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--legacy-input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()
    records = load_manifest(args.input)
    merge_audit: dict[str, object] = {}
    if args.legacy_input:
        records, merge_audit = merge_source_matched_records(
            records, load_manifest(args.legacy_input)
        )
    train, audit = select_balanced_train_records(
        [row for row in records if row.partition == "train_core"],
        train_count=args.train_count,
        seed=args.seed,
    )
    retained = train + [row for row in records if row.partition != "train_core"]
    retained.sort(key=lambda row: (row.partition, row.file_name.lower()))
    validation = validate_records(retained, require_files=True)
    audit.update({
        "pass": bool(validation["pass"]),
        "total_count": len(retained),
        "partition_counts": {
            partition: sum(row.partition == partition for row in retained)
            for partition in ("train_core", "train_calib", "train_audit", "test")
        },
        "validation_errors": validation["errors"],
        "source_matched_merge": merge_audit,
    })
    if not audit["pass"]:
        raise RuntimeError(f"balanced pilot manifest failed validation: {audit}")
    write_manifest(args.output, retained)
    Path(args.audit_output).write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False))


if __name__ == "__main__":
    main()
