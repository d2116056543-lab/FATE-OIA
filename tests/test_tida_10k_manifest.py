from fate_oia.engine.build_tida_10k_manifest import (
    assign_dual_evaluation_groups,
    attach_labels,
    source_row_to_record,
    apply_repair_overlay,
    filter_source_quality,
)
from fate_oia.engine.build_tida_balanced_pilot_manifest import (
    merge_source_matched_records,
    select_balanced_train_records,
    source_domain,
)
from fate_oia.engine.build_tida_source_complete_manifest import build_source_complete_records
from fate_oia.datasets.tida_clip_manifest import TIDAClipRecord


def _row(name, source):
    return {"file_name": name, "source_video_id": source}


def test_dual_evaluation_keeps_legacy_sources_out_of_training():
    rows = [
        _row("legacy-extra.jpg", "video-a"),
        _row("b1.jpg", "video-b"),
        _row("b2.jpg", "video-b"),
        _row("c.jpg", "video-c"),
        _row("d.jpg", "video-d"),
    ]
    result = assign_dual_evaluation_groups(
        rows,
        legacy_test_source_ids={"video-a"},
        expanded_test_count=2,
        calib_count=1,
        audit_count=1,
        seed=17,
    )

    assert [row["file_name"] for row in result["legacy_overlap_excluded"]] == ["legacy-extra.jpg"]
    assert len(result["expanded_test"]) == 2
    assert len(result["train_calib"]) == 1
    assert len(result["train_audit"]) == 1
    train_sources = {
        row["source_video_id"]
        for name in ("train_core", "train_calib", "train_audit")
        for row in result[name]
    }
    test_sources = {row["source_video_id"] for row in result["expanded_test"]} | {"video-a"}
    assert train_sources.isdisjoint(test_sources)


def test_group_partition_never_splits_same_source():
    rows = [_row(f"x{i}.jpg", "same") for i in range(3)] + [_row("y.jpg", "other")]
    result = assign_dual_evaluation_groups(
        rows,
        legacy_test_source_ids=set(),
        expanded_test_count=3,
        calib_count=1,
        audit_count=0,
        seed=3,
    )
    assert {row["source_video_id"] for row in result["expanded_test"]} == {"same"}
    assert {row["source_video_id"] for row in result["train_calib"]} == {"other"}


def test_attach_labels_creates_canonical_file_name():
    row = {"image_name": "Folder/A.JPG", "stem": "video-a"}
    labels = {"a.jpg": ((1.0, 0.0, 0.0, 0.0), tuple([1.0] + [0.0] * 20))}

    repaired = attach_labels(row, labels)

    assert repaired["image_name"] == "Folder/A.JPG"
    assert repaired["file_name"] == "A.JPG"
    assert repaired["source_video_id"] == "video-a"


def test_source_row_converts_to_tida_loader_contract(tmp_path):
    target = tmp_path / "target.jpg"
    clip = tmp_path / "clip.mp4"
    target.write_bytes(b"image")
    clip.write_bytes(b"clip")
    row = {
        "split": "train",
        "file_name": "a.jpg",
        "image_name": "a.jpg",
        "last_frame_image_path": str(target),
        "clip_path": str(clip),
        "source_video_id": "video-a",
        "action_4": [1, 0, 0, 1],
        "reason_21": [1] + [0] * 20,
        "match": {"fps": 30.0},
        "clip": {"frames_written": 151},
        "source_manifest_path": str(tmp_path / "source.jsonl"),
        "source_row_index": 4,
    }

    record = source_row_to_record(row, "train_core", compute_hashes=False)

    assert record.partition == "train_core"
    assert record.num_frames == 151
    assert record.target_frame_index == 150
    assert record.target_timestamp_seconds == 5.0
    assert record.action == (1.0, 0.0, 0.0, 1.0)
    assert len(record.reason) == 21


def test_repair_overlay_replaces_only_the_matching_clip():
    rows = [
        {"image_name": "a.jpg", "clip_path": "bad.mp4", "clip": {"frames_written": 0}, "match": {"fps": 30}},
        {"image_name": "b.jpg", "clip_path": "good.mp4", "clip": {"frames_written": 151}, "match": {"fps": 30}},
    ]
    repaired = apply_repair_overlay(rows, {
        "a.jpg": {"clip_path": "repaired.mp4", "frames_written": 150, "fps": 29.9}
    })

    assert repaired[0]["clip_path"] == "repaired.mp4"
    assert repaired[0]["clip"]["frames_written"] == 150
    assert repaired[0]["match"]["fps"] == 29.9
    assert repaired[1] == rows[1]


def test_source_quality_rejects_bad_alignment_and_zero_frames():
    rows = [
        {"image_name": "good.jpg", "match": {"mse": 0.001}, "clip": {"frames_written": 151}},
        {"image_name": "bad.jpg", "match": {"mse": 0.01}, "clip": {"frames_written": 151}},
        {"image_name": "zero.jpg", "match": {"mse": 0.001}, "clip": {"frames_written": 0}},
    ]

    accepted, rejected = filter_source_quality(rows, max_mse=0.0021)

    assert [row["image_name"] for row in accepted] == ["good.jpg"]
    assert rejected[0]["quality_rejection_reasons"] == ["endpoint_mse"]
    assert rejected[1]["quality_rejection_reasons"] == ["zero_frame_clip"]


def test_balanced_pilot_selection_covers_sources_and_action_sets(tmp_path):
    rows = []
    for batch in (3, 4, 5):
        for index in range(12):
            action = [0.0] * 4
            action[index % 4] = 1.0
            rows.append(TIDAClipRecord.from_dict({
                "official_split": "train",
                "partition": "train_core",
                "file_name": f"b{batch}-{index}.jpg",
                "target_image_path": str(tmp_path / f"b{batch}-{index}.jpg"),
                "clip_path": str(tmp_path / f"batch{batch}" / f"b{batch}-{index}.mp4"),
                "source_video_id": f"b{batch}-{index}",
                "duration_seconds": 5.0,
                "fps": 30.0,
                "num_frames": 151,
                "target_timestamp_seconds": 5.0,
                "target_frame_index": 150,
                "action": action,
                "reason": [0.0] * 21,
            }))

    selected, audit = select_balanced_train_records(rows, train_count=18, seed=7)

    assert audit["source_counts"] == {"batch3": 6, "batch4": 6, "batch5": 6}
    assert len(selected) == 18
    for batch in ("batch3", "batch4", "batch5"):
        action_sets = {tuple(row.action) for row in selected if batch in str(row.clip_path)}
        assert len(action_sets) >= 3


def test_source_domain_recognizes_all_five_video_batches(tmp_path):
    expected = {
        "bdd_oia_1000_train_test": "batch1",
        "bdd_oia_linxxx3_batch2": "batch2",
        "bdd_oia_linxxx3_batch3": "batch3",
        "bdd_oia_linxxx3_batch4_4000": "batch4",
        "bdd_oia_linxxx3_batch5_4000": "batch5",
    }
    for source_batch, domain in expected.items():
        row = TIDAClipRecord.from_dict({
            "official_split": "train", "partition": "train_core",
            "file_name": f"{domain}.jpg", "target_image_path": str(tmp_path / f"{domain}.jpg"),
            "clip_path": str(tmp_path / source_batch / f"{domain}.mp4"),
            "source_video_id": domain, "duration_seconds": 5.0, "fps": 30.0,
            "num_frames": 151, "target_timestamp_seconds": 5.0,
            "target_frame_index": 150, "action": [1, 0, 0, 0], "reason": [0] * 21,
            "source_batch": source_batch,
        })
        assert source_domain(row) == domain


def test_source_matched_merge_adds_batch1_and_batch2_without_test_leakage(tmp_path):
    def record(name, batch, partition, source):
        return TIDAClipRecord.from_dict({
            "official_split": "train", "partition": partition,
            "file_name": name, "target_image_path": str(tmp_path / name),
            "clip_path": str(tmp_path / batch / f"{name}.mp4"),
            "source_video_id": source, "duration_seconds": 5.0, "fps": 30.0,
            "num_frames": 151, "target_timestamp_seconds": 5.0,
            "target_frame_index": 150, "action": [1, 0, 0, 0], "reason": [0] * 21,
            "source_batch": batch,
        })

    expanded = [
        record("fixed.jpg", "bdd_oia_1000_train_test", "test", "fixed-source"),
        record("b3.jpg", "bdd_oia_linxxx3_batch3", "train_core", "b3-source"),
    ]
    legacy = [
        record("b1.jpg", "bdd_oia_1000_train_test", "train_core", "b1-source"),
        record("b2.jpg", "bdd_oia_linxxx3_batch2", "train_calib", "b2-source"),
        record("leak.jpg", "bdd_oia_1000_train_test", "train_core", "fixed-source"),
        record("existing-source.jpg", "bdd_oia_1000_train_test", "train_core", "b3-source"),
        record("duplicate-b3.jpg", "bdd_oia_linxxx3_batch3", "train_core", "legacy-b3"),
    ]

    merged, audit = merge_source_matched_records(expanded, legacy)

    assert {row.file_name for row in merged} == {"fixed.jpg", "b3.jpg", "b1.jpg", "b2.jpg"}
    assert audit["added_source_counts"] == {"batch1": 1, "batch2": 1}
    assert audit["excluded_test_source_overlap"] == 1
    assert audit["excluded_existing_source_overlap"] == 1
    assert audit["excluded_nonlegacy_domain"] == 1
    train_sources = {row.source_video_id for row in merged if row.partition != "test"}
    test_sources = {row.source_video_id for row in merged if row.partition == "test"}
    assert train_sources.isdisjoint(test_sources)


def test_source_complete_manifest_keeps_all_existing_rows_and_legal_legacy_train(tmp_path):
    def record(name, batch, partition, source):
        return TIDAClipRecord.from_dict({
            "official_split": "test" if partition == "test" else "train",
            "partition": partition, "file_name": name,
            "target_image_path": str(tmp_path / name),
            "clip_path": str(tmp_path / batch / f"{name}.mp4"),
            "source_video_id": source, "duration_seconds": 5.0, "fps": 30.0,
            "num_frames": 151, "target_timestamp_seconds": 5.0,
            "target_frame_index": 150, "action": [1, 0, 0, 0],
            "reason": [0] * 21, "source_batch": batch,
        })

    primary = [
        record("b3-core.jpg", "bdd_oia_linxxx3_batch3", "train_core", "b3-core"),
        record("fixed.jpg", "bdd_oia_1000_train_test", "test", "fixed_prev5s"),
    ]
    legacy = [
        record("b1-core.jpg", "bdd_oia_1000_train_test", "train_core", "b1-core"),
        record("b2-audit.jpg", "bdd_oia_linxxx3_batch2", "train_audit", "b2-audit"),
        record("same-train-source.jpg", "bdd_oia_1000_train_test", "train_audit", "b3-core"),
        record("leak.jpg", "bdd_oia_1000_train_test", "train_core", "fixed"),
    ]

    merged, audit = build_source_complete_records(primary, legacy)

    assert {row.file_name for row in merged} == {
        "b3-core.jpg", "fixed.jpg", "b1-core.jpg", "b2-audit.jpg",
        "same-train-source.jpg",
    }
    assert audit["pass"] is True
    assert audit["partition_counts"] == {
        "train_core": 3, "train_calib": 0, "train_audit": 1, "test": 1,
    }
    assert audit["source_matched_merge"]["excluded_test_source_overlap"] == 1
    assert audit["source_matched_merge"]["reassigned_existing_source_partition"] == 1
    assert audit["train_test_source_overlap"] == 0


def test_balanced_selection_redistributes_a_capacity_limited_domain(tmp_path):
    rows = []
    capacities = {"batch1": 10, "batch2": 2, "batch3": 10, "batch4": 10, "batch5": 10}
    source_names = {
        "batch1": "bdd_oia_1000_train_test", "batch2": "bdd_oia_linxxx3_batch2",
        "batch3": "bdd_oia_linxxx3_batch3", "batch4": "bdd_oia_linxxx3_batch4_4000",
        "batch5": "bdd_oia_linxxx3_batch5_4000",
    }
    for domain, capacity in capacities.items():
        for index in range(capacity):
            rows.append(TIDAClipRecord.from_dict({
                "official_split": "train", "partition": "train_core",
                "file_name": f"{domain}-{index}.jpg",
                "target_image_path": str(tmp_path / f"{domain}-{index}.jpg"),
                "clip_path": str(tmp_path / source_names[domain] / f"{index}.mp4"),
                "source_video_id": f"{domain}-{index}", "duration_seconds": 5.0,
                "fps": 30.0, "num_frames": 151, "target_timestamp_seconds": 5.0,
                "target_frame_index": 150, "action": [float(index % 2), 1, 0, 0],
                "reason": [0] * 21, "source_batch": source_names[domain],
            }))

    selected, audit = select_balanced_train_records(rows, train_count=20, seed=11)

    assert len(selected) == 20
    assert audit["source_counts"]["batch2"] == 2
    assert sum(audit["source_counts"].values()) == 20
    assert max(audit["source_counts"][key] for key in capacities if key != "batch2") <= 5
