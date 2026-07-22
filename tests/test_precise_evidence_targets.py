from pathlib import Path

import torch

from fate_oia.datasets.bdd100k_task_aware_index import TaskAwareGroundingRecord
from fate_oia.datasets.precise_grounding_adapter import PRECISEGroundingAdapter
from fate_oia.utils.precise_schema import load_evidence_fields


ROOT = Path(__file__).resolve().parents[1]


def _adapter():
    fields = load_evidence_fields(ROOT / "configs" / "precise_evidence_fields.yaml")
    return PRECISEGroundingAdapter(fields)


def test_traffic_light_without_color_is_presence_only():
    adapter = _adapter()
    record = TaskAwareGroundingRecord(("det.json",), (), (), (), {"detection": True, "lane": False, "drivable": False})
    targets = adapter.from_metadata(record, {"detection": [{"category": "traffic light", "box2d": {"x1": 1, "y1": 1, "x2": 2, "y2": 2}}]})
    light = targets["traffic_light"]
    assert light["presence"].item() == 1
    assert light["presence_valid"].item() == 1
    assert light["state_valid"].item() == 0
    assert light["state"].sum().item() == 0


def test_complete_source_without_center_actor_is_reliable_negative():
    adapter = _adapter()
    record = TaskAwareGroundingRecord(("det.json",), (), (), (), {"detection": True, "lane": False, "drivable": False})
    targets = adapter.from_metadata(record, {"detection": []})
    actor = targets["actor_center"]
    assert actor["presence"].item() == 0
    assert actor["presence_valid"].item() == 1
    assert actor["observability"].item() == 1


def test_missing_source_is_masked_not_cast_to_negative():
    adapter = _adapter()
    record = TaskAwareGroundingRecord((), (), (), (), {"detection": False, "lane": False, "drivable": False})
    targets = adapter.from_metadata(record, {})
    assert targets["actor_center"]["presence_valid"].item() == 0
    assert targets["actor_center"]["presence"].item() == 0
    assert targets["actor_center"]["reliability"].item() == 0


def test_targets_expose_required_provenance_fields():
    adapter = _adapter()
    record = TaskAwareGroundingRecord((), (), (), (), {"detection": False, "lane": False, "drivable": False})
    target = adapter.from_metadata(record, {})["traffic_sign"]
    assert {"target", "valid_mask", "reliability", "source_id", "geometry_valid", "observability_target"} <= set(target)
    assert all(torch.is_tensor(value) for key, value in target.items() if key not in {"source_id"})


def test_label_objects_and_drivable_map_are_not_silently_dropped(tmp_path: Path):
    """BDD100K stores labels in frame.objects and drivable supervision in a map."""
    from PIL import Image

    adapter = _adapter()
    map_path = tmp_path / "frame_drivable_color.png"
    Image.new("RGB", (12, 12), color=(255, 0, 0)).save(map_path)
    record = TaskAwareGroundingRecord(
        ("labels.json",), ("labels.json",), (str(map_path),), (),
        {"detection": True, "lane": True, "drivable": True, "semantic": False},
    )
    metadata = {"detection": [{"frames": [{"objects": [
        {"category": "car", "box2d": {"x1": 500, "x2": 700, "y1": 300, "y2": 500}},
        {"category": "lane/road curb", "attributes": {"style": "solid"}, "poly2d": [{"vertices": [[100, 500], [200, 650]]}]},
    ]}]}]}
    target = adapter.from_metadata(record, metadata)
    assert target["actor_center"]["presence"].item() == 1
    assert target["drivable_center"]["presence"].item() > 0
    assert target["drivable_center"]["geometry_valid"].item() == 1
    assert target["boundary_left"]["presence_valid"].item() == 1
    batch = adapter.stack_batch([target])
    assert batch["state"].shape == (1, 10, 4)
    assert batch["part_coordinates"].shape == (1, 10, 8, 2)
    assert batch["part_scales"].shape == (1, 10, 8, 2)
    assert batch["part_scales"][0, 3].sum().item() > 0
    assert batch["part_valid"].sum().item() > 0
    assert batch["soft_masks"].shape == (1, 10, 45, 80)
    assert batch["soft_masks"][0, 6].sum().item() > 0


def test_named_frame_container_is_unwrapped_before_category_matching():
    adapter = _adapter()
    record = TaskAwareGroundingRecord(("labels.json",), (), (), (), {"detection": True, "lane": False, "drivable": False})
    metadata = {"detection": [{"name": "aaaaaaaa-bbbbbbbb.jpg", "labels": [
        {"category": "car", "box2d": {"x1": 500, "x2": 700, "y1": 300, "y2": 500}}
    ]}]}
    target = adapter.from_metadata(record, metadata)
    assert target["actor_center"]["presence"].item() == 1


def test_named_clip_container_unwraps_frame_objects_and_poly2d():
    adapter = _adapter()
    record = TaskAwareGroundingRecord(("labels.json",), ("labels.json",), (), (), {"detection": True, "lane": True, "drivable": False})
    metadata = {"detection": [{"name": "aaaaaaaa-bbbbbbbb.jpg", "frames": [{"objects": [
        {"category": "car", "box2d": {"x1": 500, "x2": 700, "y1": 300, "y2": 500}},
        {"category": "lane/road curb", "poly2d": [{"vertices": [[100, 500], [200, 650]]}]},
    ]}]}]}
    target = adapter.from_metadata(record, metadata)
    assert target["actor_center"]["presence"].item() == 1
    assert target["boundary_left"]["presence"].item() == 1


def test_boundary_polylines_are_partitioned_around_ego_center_not_image_thirds():
    adapter = _adapter()
    record = TaskAwareGroundingRecord(("labels.json",), ("labels.json",), (), (), {"detection": True, "lane": True, "drivable": False})
    metadata = {"lane": [
        {"category": "lane/single white", "poly2d": [{"vertices": [[500, 400], [560, 700]]}]},
        {"category": "lane/single white", "poly2d": [{"vertices": [[720, 400], [780, 700]]}]},
    ]}
    target = adapter.from_metadata(record, metadata)
    assert target["boundary_left"]["presence"].item() == 1
    assert target["boundary_right"]["presence"].item() == 1


def test_native_bdd100k_direct_poly2d_points_produce_ordered_curve_targets():
    adapter = _adapter()
    record = TaskAwareGroundingRecord(("labels.json",), ("labels.json",), (), (), {"detection": True, "lane": True, "drivable": False})
    metadata = {"lane": [
        {"category": "lane/single white", "poly2d": [[500, 300, "L"], [520, 500, "L"], [560, 700, "L"]]},
        {"category": "lane/single white", "poly2d": [[720, 300, "L"], [740, 500, "L"], [780, 700, "L"]]},
    ]}
    target = adapter.from_metadata(record, metadata)
    for name in ("boundary_left", "boundary_right"):
        assert target[name]["presence"].item() == 1
        assert target[name]["geometry_valid"].item() == 1
        assert target[name]["part_valid"].item() == 1
        assert target[name]["soft_mask"].sum().item() > 0
        assert torch.all(target[name]["part_coordinates"][1:, 1] >= target[name]["part_coordinates"][:-1, 1])


def test_known_vehicle_is_not_mislabeled_as_other_actor_type():
    adapter = _adapter()
    record = TaskAwareGroundingRecord(("det.json",), (), (), (), {"detection": True, "lane": False, "drivable": False})
    targets = adapter.from_metadata(record, {"detection": [
        {"category": "car", "box2d": {"x1": 500, "x2": 700, "y1": 300, "y2": 500}}
    ]})
    assert torch.equal(targets["actor_center"]["state"], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_coverage_reports_state_and_source_completeness():
    adapter = _adapter()
    complete = TaskAwareGroundingRecord(("det.json",), (), (), (), {"detection": True, "lane": False, "drivable": False})
    missing = TaskAwareGroundingRecord((), (), (), (), {"detection": False, "lane": False, "drivable": False})
    report = adapter.coverage([
        adapter.from_metadata(complete, {"detection": [{"category": "car", "box2d": {"x1": 500, "x2": 700, "y1": 300, "y2": 500}}]}),
        adapter.from_metadata(missing, {}),
    ])
    center = report["actor_center"]
    assert center["state_valid_count"] == 1
    assert center["source_complete_rate"] == 0.5
