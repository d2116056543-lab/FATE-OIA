import json
from pathlib import Path

from fate_oia.datasets.bdd100k_task_aware_index import BDD100KTaskAwareIndex


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_same_stem_keeps_detection_lane_and_drivable_sources(tmp_path: Path):
    root = tmp_path / "bdd100k"
    _write(root / "labels" / "det" / "scene_a.json", {"name": "scene_a.jpg", "frames": []})
    _write(root / "labels" / "lane" / "scene_a.json", {"name": "scene_a.jpg", "lanes": []})
    (root / "drivable" / "scene_a.png").parent.mkdir(parents=True, exist_ok=True)
    (root / "drivable" / "scene_a.png").write_bytes(b"not-a-real-png")
    index = BDD100KTaskAwareIndex(root, emit_manifest=False)
    record = index.get("scene_a.jpg")
    assert len(record.detection_jsons) == 1
    assert len(record.lane_jsons) == 1
    assert len(record.drivable_maps) == 1
    assert record.source_complete["detection"] is True
    assert record.source_complete["lane"] is True
    assert record.source_complete["drivable"] is True


def test_missing_source_is_unknown_not_negative(tmp_path: Path):
    root = tmp_path / "bdd100k"
    _write(root / "labels" / "det" / "only_det.json", {"name": "only_det.jpg", "frames": []})
    index = BDD100KTaskAwareIndex(root, emit_manifest=False)
    record = index.get("only_det.jpg")
    assert record.source_complete["detection"] is True
    assert record.source_complete["lane"] is False
    assert record.source_complete["drivable"] is False


def test_bdd_oia_clip_frame_suffix_resolves_to_bdd100k_base_stem(tmp_path: Path):
    root = tmp_path / "bdd100k"
    _write(root / "labels" / "det" / "9af44b54-7aae6c7d.json", {"name": "9af44b54-7aae6c7d.jpg", "frames": []})
    index = BDD100KTaskAwareIndex(root, emit_manifest=False)
    assert index.get("9af44b54-7aae6c7d_3.jpg").source_complete["detection"] is True
    assert index.metadata_for("9af44b54-7aae6c7d_1.jpg")["detection"]


def test_numeric_suffix_is_only_removed_from_uuid_style_bdd_clip_stems(tmp_path: Path):
    root = tmp_path / "bdd100k"
    _write(root / "labels" / "det" / "ordinary_3.json", {"name": "ordinary_3.jpg", "frames": []})
    index = BDD100KTaskAwareIndex(root, emit_manifest=False)
    assert index.get("ordinary_3.jpg").source_complete["detection"] is True
    assert index.get("ordinary_1.jpg").source_complete["detection"] is False


def test_multiframe_json_metadata_is_bound_to_its_own_stem_without_cross_image_leakage(tmp_path: Path):
    root = tmp_path / "bdd100k"
    payload = [
        {"name": "aaaaaaaa-bbbbbbbb.jpg", "labels": [{"category": "car", "box2d": {"x1": 1, "y1": 2, "x2": 3, "y2": 4}}]},
        {"name": "cccccccc-dddddddd.jpg", "labels": [{"category": "pedestrian", "box2d": {"x1": 5, "y1": 6, "x2": 7, "y2": 8}}]},
    ]
    _write(root / "bdd100k_labels" / "det_train.json", payload)
    index = BDD100KTaskAwareIndex(root, emit_manifest=False)
    first = index.metadata_for("aaaaaaaa-bbbbbbbb_1.jpg")["detection"]
    second = index.metadata_for("cccccccc-dddddddd_3.jpg")["detection"]
    assert first == [payload[0]]
    assert second == [payload[1]]


def test_poly2d_in_list_format_is_indexed_as_lane_for_only_that_frame(tmp_path: Path):
    root = tmp_path / "bdd100k"
    payload = [
        {"name": "aaaaaaaa-bbbbbbbb.jpg", "labels": [{"category": "lane", "poly2d": [{"vertices": [[0, 0], [1, 1]]}]}]},
        {"name": "cccccccc-dddddddd.jpg", "labels": []},
    ]
    _write(root / "bdd100k_labels" / "labels.json", payload)
    index = BDD100KTaskAwareIndex(root, emit_manifest=False)
    assert index.get("aaaaaaaa-bbbbbbbb_1.jpg").source_complete["lane"] is True
    assert index.get("cccccccc-dddddddd_1.jpg").source_complete["lane"] is False


def test_manifest_has_counts_duplicates_missing_and_hashes(tmp_path: Path):
    root = tmp_path / "bdd100k"
    _write(root / "detection" / "a.json", {"name": "a.jpg", "frames": []})
    manifest = tmp_path / "manifest.json"
    index = BDD100KTaskAwareIndex(root, manifest_path=manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["source_counts"]["detection"] == 1
    assert payload["paths"]["detection"]
    assert payload["hashes"]
    assert index.get("missing.jpg").source_complete["detection"] is False


def test_metadata_is_preparsed_once_at_startup(tmp_path: Path):
    root = tmp_path / "bdd100k"
    _write(root / "detection" / "a.json", {"name": "a.jpg", "labels": [{"category": "car"}]})
    index = BDD100KTaskAwareIndex(root, emit_manifest=False)
    first = index.metadata_for("a.jpg")
    second = index.metadata_for("a.jpg")
    assert first is second
    assert first["detection"]


def test_malformed_json_is_recorded_and_never_becomes_a_negative_source(tmp_path: Path):
    root = tmp_path / "bdd100k"
    bad = root / "labels" / "broken.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not-json", encoding="utf-8")
    index = BDD100KTaskAwareIndex(root, emit_manifest=False)
    assert index.get("broken.jpg").source_complete["detection"] is False
    assert index.invalid_json_paths == [str(bad)]
