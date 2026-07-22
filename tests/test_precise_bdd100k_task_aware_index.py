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
