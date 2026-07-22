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
