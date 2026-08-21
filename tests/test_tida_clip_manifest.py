from __future__ import annotations

from pathlib import Path

import pytest

from fate_oia.datasets.tida_clip_manifest import (
    TIDAClipRecord,
    partition_source_grouped_records,
    partition_train_records,
    validate_records,
)


def _row(i: int, source: str | None = None) -> TIDAClipRecord:
    return TIDAClipRecord(
        official_split="train", partition="unassigned", file_name=f"{i}.jpg",
        target_image_path=Path(f"/{i}.jpg"), clip_path=Path(f"/{i}.mp4"),
        source_video_id=source or f"source-{i}", duration_seconds=5.0, fps=30.0,
        num_frames=151, target_timestamp_seconds=0.0, target_frame_index=150,
        action=(1.0, 0.0, 0.0, 0.0), reason=(0.0,) * 21,
    )


def test_partition_is_exact_deterministic_and_group_safe():
    rows = [_row(i) for i in range(3115)]
    first = partition_train_records(rows, seed=20260821, calib_count=312, audit_count=512)
    second = partition_train_records(list(reversed(rows)), seed=20260821, calib_count=312, audit_count=512)
    assert {r.file_name: r.partition for r in first} == {r.file_name: r.partition for r in second}
    counts = {p: sum(r.partition == p for r in first) for p in ("train_core", "train_calib", "train_audit")}
    assert counts == {"train_core": 2291, "train_calib": 312, "train_audit": 512}
    assert validate_records(first, require_files=False)["pass"]


def test_formal_manifest_rejects_legacy_split_field(tmp_path):
    with pytest.raises(ValueError, match="legacy split"):
        TIDAClipRecord.from_dict({"split": "train"})


def test_source_grouped_partition_preserves_rows_and_prevents_official_split_leakage():
    first = _row(0, source="shared-source")
    second = _row(1, source="shared-source")
    rows = [
        TIDAClipRecord(**{**first.__dict__, "official_split": "train"}),
        TIDAClipRecord(**{**second.__dict__, "official_split": "test"}),
    ]
    for index in range(2, 4000):
        row = _row(index, source=f"source-{index}")
        rows.append(TIDAClipRecord(**{**row.__dict__, "official_split": "train" if index % 4 else "test"}))
    result = partition_source_grouped_records(rows, test_count=885, calib_count=312, audit_count=512)
    counts = {partition: sum(row.partition == partition for row in result) for partition in ("train_core", "train_calib", "train_audit", "test")}
    assert counts == {"train_core": 2291, "train_calib": 312, "train_audit": 512, "test": 885}
    assert len(result) == 4000
    assert validate_records(result, require_files=False)["pass"]
    owners = {}
    for row in result:
        owners.setdefault(row.source_video_id, set()).add(row.partition)
    assert all(len(partitions) == 1 for partitions in owners.values())
