"""T2/P1 behavioral contracts for task-aware BDD100K grounding.

These tests use tiny synthetic annotation files on purpose: they prove that
metadata is parsed once and that task sources are merged, without requiring a
real BDD100K install or reading visual features from disk.
"""

from __future__ import annotations

import json
import inspect
import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from fate_oia.datasets.bdd100k_task_aware_index import (
    RAELGroundingRecord,
    RAELTaskAwareBDD100KIndex,
)
from fate_oia.datasets.rael_grounding_targets import (
    aggregate_grounding_coverage,
    build_entity_grounding_targets,
    build_road_grounding_targets,
    entity_match_cost,
    hungarian_entity_matching,
)
import fate_oia.datasets.bdd100k_task_aware_index as task_index_module
import fate_oia.datasets.rael_grounding_targets as target_module
import fate_oia.transforms_rael as transform_module
from fate_oia.transforms_rael import RAELGroundingTransform


class _SpawnPickleGroundingDataset(Dataset[int]):
    """Top-level on purpose: Windows spawn must import and pickle this class."""

    def __init__(self, index: RAELTaskAwareBDD100KIndex) -> None:
        self.index = index

    def __len__(self) -> int:
        return 2

    def __getitem__(self, item: int) -> int:
        record = self.index.lookup(r"C:\bdd100k\spawn_7.jpg")
        return len(record.detections)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture()
def synthetic_sources(tmp_path: Path) -> dict[str, Path]:
    stem = "same_stem.jpg"
    detections = {
        "frames": [
            {
                "name": stem,
                "labels": [
                    {
                        "category": "car",
                        "box2d": {"x1": 10, "y1": 5, "x2": 30, "y2": 25},
                        "attributes": {},
                        "sector": "left",
                    },
                    {
                        "category": "traffic light",
                        "box2d": {"x1": 60, "y1": 4, "x2": 80, "y2": 25},
                        "attributes": {"trafficLightColor": "green"},
                        "sector": "right",
                    },
                ],
            }
        ]
    }
    lanes = {
        "frames": [
            {
                "name": stem,
                "lanes": [
                    {
                        "category": "lane",
                        "side": "left",
                        "points": [[10, 45], [20, 30], [30, 20]],
                        "attributes": {"style": "solid"},
                    }
                ],
            }
        ]
    }
    drivable = {
        "frames": [
            {
                "name": stem,
                "drivable": [
                    {
                        "category": "drivable",
                        "side": "left",
                        "polygon": [[0, 49], [48, 49], [45, 20], [5, 20]],
                    }
                ],
            }
        ]
    }
    return {
        "detections": _write_json(tmp_path / "det.json", detections),
        "lanes": _write_json(tmp_path / "lane.json", lanes),
        "drivable": _write_json(tmp_path / "drive.json", drivable),
    }


def _record(*, detection_complete: bool = True, lane_complete: bool = True, drivable_complete: bool = True):
    return RAELGroundingRecord(
        detections=(
            {
                "category": "car",
                "box": [10.0, 5.0, 30.0, 25.0],
                "sector": "left",
                "attributes": {"approach_side": "left"},
            },
            {
                "category": "traffic_light",
                "box": [60.0, 4.0, 80.0, 25.0],
                "sector": "right",
                "attributes": {"trafficLightColor": "green"},
            },
        ),
        lanes=(
            {
                "category": "lane",
                "side": "left",
                "points": [[10.0, 45.0], [20.0, 30.0], [30.0, 20.0]],
                "attributes": {"style": "solid"},
            },
        ),
        drivable=(
            {
                "category": "drivable",
                "side": "left",
                "polygon": [[0.0, 49.0], [48.0, 49.0], [45.0, 20.0], [5.0, 20.0]],
            },
        ),
        source_complete={
            "detections": detection_complete,
            "lanes": lane_complete,
            "drivable": drivable_complete,
        },
    )


def test_record_is_frozen_and_index_merges_all_task_sources_once(synthetic_sources: dict[str, Path]) -> None:
    index = RAELTaskAwareBDD100KIndex(**synthetic_sources)
    record = index.lookup("same_stem.jpg")
    assert isinstance(record, RAELGroundingRecord)
    assert len(record.detections) == 2
    assert len(record.lanes) == 1
    assert len(record.drivable) == 1
    assert record.source_complete == {"detections": True, "lanes": True, "drivable": True}
    with pytest.raises(FrozenInstanceError):
        record.detections = ()  # type: ignore[misc]

    # Corrupting a source after construction must not affect lookup: batch code
    # reads pre-parsed metadata, never re-parses JSON or drops another source.
    synthetic_sources["detections"].write_text("not-json", encoding="utf-8")
    again = index.lookup("same_stem.jpg")
    assert again == record
    manifest = index.manifest()
    assert manifest["parse_calls"] == {"detections": 1, "lanes": 1, "drivable": 1}
    assert manifest["source_hashes"]["detections"]
    assert manifest["coverage"]["stems"] == 1
    assert manifest["coverage"]["grounding_available"] is False
    for key in (
        "matched_entity_count",
        "unmatched_positive_count",
        "reliable_negative_count",
        "unknown_count",
        "traffic_state_valid_count",
        "drivable_valid_count",
        "boundary_valid_count",
    ):
        assert key not in manifest["coverage"]


def test_manifest_is_stable_writable_and_reduced_stem_alias_is_explicit(tmp_path: Path) -> None:
    payload = {"frames": [{"name": "alias_frame_7.jpg", "labels": [{"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}}]}]}
    det = _write_json(tmp_path / "det.json", payload)
    lane = _write_json(tmp_path / "lane.json", {"frames": [{"name": "alias_frame_7.jpg", "lanes": []}]})
    drive = _write_json(tmp_path / "drive.json", {"frames": [{"name": "alias_frame_7.jpg", "drivable": []}]})
    first = RAELTaskAwareBDD100KIndex(detections=det, lanes=lane, drivable=drive)
    second = RAELTaskAwareBDD100KIndex(detections=det, lanes=lane, drivable=drive)
    assert first.source_stem_aliases("alias_frame_7.jpg") == ("alias_frame_7", "alias_frame")
    assert len(first.lookup("alias_frame_999.jpg").detections) == 1
    entity_targets = build_entity_grounding_targets(
        [
            {"slot_id": 0, "category": "car", "box": [0, 0, 5, 5], "sector": "left"},
        ],
        RAELGroundingRecord(
            detections=({"category": "car", "box": [0, 0, 5, 5], "sector": "left", "attributes": {}},),
            lanes=(),
            drivable=(),
            source_complete={"detections": True, "lanes": False, "drivable": False},
        ),
        image_size=(10, 10),
    )
    road_targets = build_road_grounding_targets(
        RAELGroundingRecord(
            detections=(),
            lanes=({"category": "lane", "side": "left", "points": [[1, 1], [2, 2]]},),
            drivable=({"category": "drivable", "side": "left", "polygon": [[0, 9], [2, 9], [1, 2]]},),
            source_complete={"detections": False, "lanes": True, "drivable": True},
        )
    )
    aggregate = aggregate_grounding_coverage([entity_targets], [road_targets])
    first_manifest = first.manifest(grounding_coverage=aggregate)
    second_manifest = second.manifest(grounding_coverage=aggregate)
    assert first_manifest["manifest_hash"] == second_manifest["manifest_hash"]
    assert first_manifest["coverage"]["grounding_available"] is True
    assert first_manifest["coverage"]["matched_entity_count"] == 1
    assert first_manifest["coverage"]["drivable_valid_count"] == 1
    assert first_manifest["coverage"]["boundary_valid_count"] == 1
    written = first.write_manifest(tmp_path / "manifest.json", grounding_coverage=aggregate)
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == written


def test_hungarian_cost_uses_exact_contract_weights_and_assignment() -> None:
    slot = {"slot_id": 0, "category": "car", "box": [0.0, 0.0, 10.0, 10.0], "sector": "left"}
    wrong = {"category": "pedestrian", "box": [10.0, 0.0, 20.0, 10.0], "sector": "right"}
    correct = {"category": "car", "box": [0.0, 0.0, 10.0, 10.0], "sector": "left"}
    cost, parts = entity_match_cost(slot, wrong, image_size=(100, 100), return_components=True)
    assert cost == pytest.approx(
        1.0 * parts["type"] + 2.0 * parts["box_l1"] + 2.0 * parts["giou"] + 0.5 * parts["sector"]
    )
    matches = hungarian_entity_matching([slot], [wrong, correct], image_size=(100, 100))
    assert [(item.slot_index, item.detection_index) for item in matches] == [(0, 1)]


def test_unmatched_objectness_is_reliable_only_when_detection_source_is_complete() -> None:
    slots = [
        {"slot_id": 0, "category": "car", "box": [10, 5, 30, 25], "sector": "left"},
        {"slot_id": 1, "category": "pedestrian", "box": [40, 5, 50, 25], "sector": "center"},
        {"slot_id": 2, "category": "pedestrian", "box": [50, 5, 60, 25], "sector": "center"},
    ]
    incomplete = build_entity_grounding_targets(slots, _record(detection_complete=False), image_size=(100, 50))
    complete = build_entity_grounding_targets(slots, _record(detection_complete=True), image_size=(100, 50))
    # There are more slots than detections, so exactly one slot remains
    # structurally unmatched regardless of which non-car slot has lower cost.
    unknown_items = [item for item in incomplete.objectness if item.matched_detection_index is None]
    reliable_items = [item for item in complete.objectness if item.matched_detection_index is None]
    assert len(unknown_items) == len(reliable_items) == 1
    unknown, reliable = unknown_items[0], reliable_items[0]
    assert unknown.target == 0.0 and unknown.reliable is False
    assert reliable.target == 0.0 and reliable.reliable is True
    assert incomplete.coverage["reliable_negative_count"] == 0
    assert incomplete.coverage["unknown_count"] >= 1
    assert complete.coverage["reliable_negative_count"] >= 1


def test_traffic_state_targets_bind_detection_and_matched_slot_and_reject_unknown_color() -> None:
    slots = [
        {"slot_id": 0, "category": "car", "box": [10, 5, 30, 25], "sector": "left"},
        {"slot_id": 1, "category": "traffic_light", "box": [60, 4, 80, 25], "sector": "right"},
    ]
    valid = build_entity_grounding_targets(slots, _record(), image_size=(100, 50))
    assert len(valid.traffic_state_targets) == 1
    target = valid.traffic_state_targets[0]
    assert target.detection_index == 1 and target.matched_slot_index == 1
    assert target.state == "green" and target.valid is True
    assert valid.coverage["traffic_state_valid_count"] == 1
    unknown_record = RAELGroundingRecord(
        detections=tuple({**item, "attributes": {"trafficLightColor": "unknown"}} if item["category"] == "traffic_light" else item for item in _record().detections),
        lanes=_record().lanes,
        drivable=_record().drivable,
        source_complete=_record().source_complete,
    )
    invalid = build_entity_grounding_targets(slots, unknown_record, image_size=(100, 50))
    assert invalid.traffic_state_targets[0].valid is False
    assert invalid.coverage["traffic_state_valid_count"] == 0


def test_road_targets_have_valid_masks_counts_and_disable_empty_boundary_loss() -> None:
    populated = build_road_grounding_targets(_record())
    assert populated.coverage["drivable_valid_count"] == 1
    assert populated.coverage["boundary_valid_count"] == 1
    assert populated.drivable_valid_mask == (True, False, False)
    assert populated.boundary_valid_mask == (True, False)
    assert populated.active_boundary_loss is True
    empty_lane = RAELGroundingRecord(
        detections=(), lanes=(), drivable=_record().drivable,
        source_complete={"detections": True, "lanes": True, "drivable": True},
    )
    empty = build_road_grounding_targets(empty_lane)
    assert empty.coverage["boundary_valid_count"] == 0
    assert empty.active_boundary_loss is False


def test_canonical_and_mirror_transform_image_and_all_grounding_geometry_together() -> None:
    image = Image.new("RGB", (100, 50), color=(0, 0, 0))
    image.putpixel((10, 10), (255, 0, 0))
    transform = RAELGroundingTransform(image_height=360, image_width=640, normalize=False)
    canonical = transform(image, _record(), mirror=False)
    mirrored = transform(image, _record(), mirror=True)
    assert canonical.image.shape == (3, 360, 640)
    assert torch.allclose(mirrored.image, torch.flip(canonical.image, dims=[2]))
    assert canonical.record.detections[0]["box"] == pytest.approx([64.0, 52.0, 192.0, 180.0])
    assert mirrored.record.detections[0]["box"] == pytest.approx([448.0, 52.0, 576.0, 180.0])
    assert canonical.record.detections[0]["sector"] == "left"
    assert mirrored.record.detections[0]["sector"] == "right"
    assert mirrored.record.detections[0]["attributes"]["approach_side"] == "right"
    assert mirrored.record.lanes[0]["side"] == "right"
    assert mirrored.record.drivable[0]["side"] == "right"
    assert mirrored.record.lanes[0]["points"][0][0] == pytest.approx(576.0)
    assert mirrored.record.drivable[0]["polygon"][0][0] == pytest.approx(640.0)


def test_mirror_changes_only_controlled_direction_tokens() -> None:
    record = _record()
    car = {**record.detections[0], "attributes": {"approach_side": "left", "lighting": "bright", "pose": "upright"}}
    controlled = RAELGroundingRecord(
        detections=(car,), lanes=record.lanes, drivable=record.drivable, source_complete=record.source_complete,
    )
    result = RAELGroundingTransform(image_height=360, image_width=640, normalize=False)(
        Image.new("RGB", (100, 50)), controlled, mirror=True
    )
    attributes = result.record.detections[0]["attributes"]
    assert attributes["approach_side"] == "right"
    assert attributes["lighting"] == "bright"
    assert attributes["pose"] == "upright"


def test_mirror_projects_bdd_poly2d_vertices_recursively() -> None:
    structured_lane = {
        "category": "lane",
        "side": "left",
        "points": [{"vertices": [[10.0, 45.0], [20.0, 30.0]]}],
        "attributes": {"style": "solid"},
    }
    record = RAELGroundingRecord(
        detections=(),
        lanes=(structured_lane,),
        drivable=(),
        source_complete={"detections": False, "lanes": True, "drivable": False},
    )
    image = Image.new("RGB", (100, 50), color=(0, 0, 0))
    mirrored = RAELGroundingTransform(image_height=360, image_width=640, normalize=False)(image, record, mirror=True)
    assert mirrored.record.lanes[0]["points"][0]["vertices"][0][0] == pytest.approx(576.0)


def test_grounding_modules_forbid_semantic_segmentation_dependency() -> None:
    source = (
        inspect.getsource(task_index_module)
        + inspect.getsource(target_module)
        + inspect.getsource(transform_module)
    )
    assert "semantic_seg" not in source
    assert "bdd100k_seg" not in source


def test_record_is_deeply_immutable_and_mutable_copy_cannot_pollute_lookup(tmp_path: Path) -> None:
    payload = {
        "frames": [{
            "name": "safe.jpg",
            "labels": [{
                "category": "car",
                "box2d": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
                "attributes": {"nested": {"values": [1, 2]}},
            }],
        }],
    }
    index = RAELTaskAwareBDD100KIndex(detections=_write_json(tmp_path / "det.json", payload))
    record = index.lookup("safe.jpg")
    with pytest.raises(TypeError):
        record.detections[0]["attributes"]["nested"]["values"][0] = 99
    safe = record.mutable_copy()
    safe["detections"][0]["attributes"]["nested"]["values"][0] = 99
    again = index.lookup("safe.jpg")
    assert again.detections[0]["attributes"]["nested"]["values"] == (1, 2)


def test_reduced_stem_union_merges_cross_source_aliases_deterministically(tmp_path: Path) -> None:
    det = _write_json(tmp_path / "det.json", {"frames": [{"name": "clip.jpg", "labels": [{"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}}]}]})
    lane = _write_json(tmp_path / "lane.json", {"frames": [{"name": "clip_7.jpg", "lanes": [{"side": "left", "points": [[0, 1], [1, 0]]}]}]})
    drive = _write_json(tmp_path / "drive.json", {"frames": [{"name": "clip.jpg", "drivable": [{"side": "left", "polygon": [[0, 2], [2, 2], [1, 1]]}]}]})
    index = RAELTaskAwareBDD100KIndex(detections=det, lanes=lane, drivable=drive)
    record = index.lookup(r"C:\bdd100k\clip_7.jpg")
    assert (len(record.detections), len(record.lanes), len(record.drivable)) == (1, 1, 1)
    manifest = index.manifest()
    assert manifest["reduced_stem_policy"] == "union_all_exact_aliases_by_reduced_stem"
    assert manifest["reduced_stem_aliases"]["clip"] == ["clip", "clip_7"]


def test_empty_labels_fall_through_to_objects_with_valid_detection(tmp_path: Path) -> None:
    payload = {"frames": [{"name": "fallback.jpg", "labels": [], "objects": [{"category": "car", "box2d": {"x1": 0, "y1": 0, "x2": 5, "y2": 5}}]}]}
    index = RAELTaskAwareBDD100KIndex(detections=_write_json(tmp_path / "det.json", payload))
    assert len(index.lookup("fallback.jpg").detections) == 1


def test_invalid_boxes_are_dropped_or_fail_fast_before_hungarian(tmp_path: Path) -> None:
    bad = {"frames": [{"name": "bad.jpg", "labels": [{"category": "car", "box2d": {"x1": float("nan"), "y1": 0, "x2": 5, "y2": 5}}, {"category": "car", "box2d": {"x1": 5, "y1": 0, "x2": 5, "y2": 5}}]}]}
    index = RAELTaskAwareBDD100KIndex(detections=_write_json(tmp_path / "bad.json", bad))
    assert index.lookup("bad.jpg").detections == ()
    valid = {"category": "car", "box": [0, 0, 5, 5], "sector": "left"}
    invalid = {"category": "car", "box": [0, 0, float("inf"), 5], "sector": "left"}
    with pytest.raises(ValueError):
        entity_match_cost(valid, invalid, image_size=(10, 10))


def test_transform_deep_copies_canonical_and_mirror_and_handles_camel_case_direction() -> None:
    record = RAELGroundingRecord(
        detections=({
            "category": "car", "box": [0, 0, 5, 5], "sector": "left",
            "attributes": {"laneDirection": "LEFT", "description": "bright upright"},
        },),
        lanes=(), drivable=(),
        source_complete={"detections": True, "lanes": False, "drivable": False},
    )
    transform = RAELGroundingTransform(image_height=16, image_width=16, normalize=False)
    canonical = transform(Image.new("RGB", (10, 10)), record, mirror=False)
    mirrored = transform(Image.new("RGB", (10, 10)), record, mirror=True)
    assert canonical.record.detections[0] is not record.detections[0]
    assert canonical.record.detections[0]["attributes"] is not record.detections[0]["attributes"]
    assert mirrored.record.detections[0]["attributes"]["laneDirection"] == "RIGHT"
    assert mirrored.record.detections[0]["attributes"]["description"] == "bright upright"


def test_manifest_coverage_rejects_non_integer_values_and_writes_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = RAELTaskAwareBDD100KIndex()
    required = {
        "matched_entity_count": 1,
        "unmatched_positive_count": 0,
        "reliable_negative_count": 0,
        "unknown_count": 0,
        "traffic_state_valid_count": 0,
        "drivable_valid_count": 0,
        "boundary_valid_count": 0,
    }
    for invalid in (True, "1", 1.5):
        bad = {**required, "matched_entity_count": invalid}
        with pytest.raises((TypeError, ValueError)):
            index.manifest(grounding_coverage=bad)
    replace_calls: list[tuple[Path, Path]] = []
    original_replace = task_index_module.os.replace

    def spy_replace(source: str | Path, destination: str | Path) -> None:
        replace_calls.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(task_index_module.os, "replace", spy_replace)
    destination = tmp_path / "manifest.json"
    written = index.write_manifest(destination, grounding_coverage=required)
    assert json.loads(destination.read_text(encoding="utf-8")) == written
    assert replace_calls and replace_calls[0][1] == destination
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_record_index_pickle_round_trip_and_windows_spawn_dataloader(tmp_path: Path) -> None:
    payload = {
        "frames": [{
            "name": "spawn.jpg",
            "labels": [{
                "category": "car",
                "box2d": {"x1": 0, "y1": 0, "x2": 5, "y2": 5},
                "attributes": {"nested": {"values": [1, 2]}},
            }],
        }],
    }
    index = RAELTaskAwareBDD100KIndex(detections=_write_json(tmp_path / "spawn.json", payload))
    record = index.lookup("spawn.jpg")
    restored_record = pickle.loads(pickle.dumps(record))
    restored_index = pickle.loads(pickle.dumps(index))
    assert restored_record == record
    assert restored_index.lookup(r"C:\bdd100k\spawn_7.jpg") == record
    loader = DataLoader(
        _SpawnPickleGroundingDataset(restored_index),
        batch_size=2,
        num_workers=1,
        multiprocessing_context="spawn",
        persistent_workers=False,
        timeout=30,
    )
    assert next(iter(loader)).tolist() == [1, 1]
