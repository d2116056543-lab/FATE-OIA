import json
from pathlib import Path

from fate_oia.engine.build_tida_full_manifest import (
    discover_labeled_clips,
    partition_official_split,
    partition_official_source_disjoint,
)
from fate_oia.datasets.tida_clip_manifest import assert_no_partition_leakage
from fate_oia.datasets.tida_clip_manifest import write_manifest
from fate_oia.datasets.bdd_oia_video import BDDOIAVideoDataset


def _write_case(root: Path, split: str, stem: str, source: str, *, modern: bool, valid: bool = True):
    clip_dir = root / "clips_prev5s" / split
    label_dir = root / "labels" / split
    frame_dir = root / "last_frames" / split
    clip_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    clip = clip_dir / f"{stem}_prev5s.mp4"
    clip.write_bytes(b"video")
    (frame_dir / f"{stem}_lastframe.jpg").write_bytes(b"image")
    if modern:
        label_name = f"{stem}_prev5s.json"
    else:
        label_name = f"{stem}.json"
    payload = {
        "split": split,
        "image_name": f"{stem}.jpg",
        "video_stem": source,
        "action_4": [1, 0, 0, 0] if valid else None,
        "reason_21": [1] + [0] * 20 if valid else None,
    }
    (label_dir / label_name).write_text(json.dumps(payload), encoding="utf-8")


def test_discovery_supports_both_label_naming_generations(tmp_path):
    _write_case(tmp_path, "train", "source-a_1", "source-a", modern=True)
    _write_case(tmp_path, "test", "source-b_3", "source-b", modern=False)

    rows, audit = discover_labeled_clips(tmp_path, probe=lambda _path: (30.0, 151))

    assert len(rows) == 2
    assert audit["accepted_by_split"] == {"train": 1, "test": 1}
    assert all(len(row.action) == 4 and len(row.reason) == 21 for row in rows)
    assert all(row.target_frame_index == 150 for row in rows)


def test_discovery_rejects_present_but_null_labels(tmp_path):
    _write_case(tmp_path, "train", "source-a", "source-a", modern=False, valid=False)

    rows, audit = discover_labeled_clips(tmp_path, probe=lambda _path: (30.0, 151))

    assert rows == []
    assert audit["rejections"] == {"null_or_invalid_target": 1}


def test_discovery_retries_transient_parallel_video_probe(tmp_path):
    _write_case(tmp_path, "train", "source-a", "source-a", modern=False)
    attempts = 0

    def transient_probe(_path):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("decoder_busy")
        return 30.0, 151

    rows, audit = discover_labeled_clips(
        tmp_path, probe=transient_probe, probe_workers=2
    )

    assert len(rows) == 1
    assert audit["rejections"] == {}
    assert audit["retried_video_probes"] == 1


def test_official_test_sources_are_excluded_from_all_train_partitions(tmp_path):
    rows = []
    for split, name, source in (
        ("test", "test-a", "shared"),
        ("train", "train-leak", "shared"),
        ("train", "train-a", "source-a"),
        ("train", "train-b", "source-b"),
        ("train", "train-c", "source-c"),
    ):
        _write_case(tmp_path, split, name, source, modern=False)
    rows, _ = discover_labeled_clips(tmp_path, probe=lambda _path: (30.0, 151))

    partitioned, audit = partition_official_source_disjoint(
        rows, calib_count=1, audit_count=1, seed=7
    )

    train_sources = {row.source_video_id for row in partitioned if row.partition != "test"}
    test_sources = {row.source_video_id for row in partitioned if row.partition == "test"}
    assert train_sources.isdisjoint(test_sources)
    assert audit["excluded_train_test_source_overlap"] == 1
    assert {row.partition for row in partitioned} == {
        "train_core", "train_calib", "train_audit", "test"
    }


def test_official_split_keeps_all_valid_train_even_when_source_appears_in_test(tmp_path):
    for split, name, source in (
        ("test", "test-a", "shared"),
        ("train", "train-shared", "shared"),
        ("train", "train-a", "source-a"),
        ("train", "train-b", "source-b"),
    ):
        _write_case(tmp_path, split, name, source, modern=False)
    rows, _ = discover_labeled_clips(tmp_path, probe=lambda _path: (30.0, 151))

    partitioned, audit = partition_official_split(
        rows, calib_count=1, audit_count=1, seed=7
    )

    assert sum(row.official_split == "train" for row in partitioned) == 3
    assert sum(row.partition == "test" for row in partitioned) == 1
    assert audit["excluded_train_test_source_overlap"] == 0
    assert audit["diagnostic_cross_split_source_count"] == 1
    assert sum(audit["partition_counts"].get(name, 0) for name in (
        "train_core", "train_calib", "train_audit"
    )) == 3


def test_leakage_audit_skips_quadratic_near_duplicate_scan_without_phashes():
    class CountingRow(dict):
        endpoint_reads = 0

        def get(self, key, default=None):
            if key == "endpoint_phash":
                type(self).endpoint_reads += 1
            return super().get(key, default)

    rows = [
        CountingRow(
            partition="train_core" if index % 2 else "test",
            source_video_id=f"source-{index}",
            clip_sha256="",
        )
        for index in range(100)
    ]

    assert_no_partition_leakage(rows)

    assert CountingRow.endpoint_reads <= len(rows)


def test_training_dataset_can_consume_every_official_train_partition(tmp_path):
    for split, name, source in (
        ("test", "test-a", "test-source"),
        ("train", "train-a", "source-a"),
        ("train", "train-b", "source-b"),
        ("train", "train-c", "source-c"),
    ):
        _write_case(tmp_path, split, name, source, modern=False)
    rows, _ = discover_labeled_clips(tmp_path, probe=lambda _path: (30.0, 151))
    partitioned, _ = partition_official_split(rows, calib_count=1, audit_count=1, seed=7)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(manifest, partitioned)

    dataset = BDDOIAVideoDataset(
        manifest, ("train_core", "train_calib", "train_audit"), training=True
    )

    assert len(dataset) == 3
    assert {record.partition for record in dataset.records} == {
        "train_core", "train_calib", "train_audit"
    }
