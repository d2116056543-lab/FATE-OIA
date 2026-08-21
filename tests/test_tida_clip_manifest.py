from pathlib import Path

import pytest

from fate_oia.datasets.tida_clip_manifest import TIDAClipRecord, partition_train_records, validate_records


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
