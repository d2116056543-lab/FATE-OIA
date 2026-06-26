from __future__ import annotations

import torch

from fate_oia.models.acpr_predicate_targets import WeakPredicateTargetBuilder


def _targeted_names(payload, builder: WeakPredicateTargetBuilder, row: int = 0) -> set[str]:
    target = payload["predicate_targets"][row]
    mask = payload["predicate_mask"][row]
    return {name for name, idx in builder.name_to_id.items() if float(target[idx]) > 0.5 and float(mask[idx]) > 0.5}


def test_weak_predicates_do_not_turn_unknown_states_positive():
    builder = WeakPredicateTargetBuilder("configs/acpr_scene_predicates.yaml")
    record = {
        "frames": [
            {
                "labels": [
                    {"category": "traffic light", "box2d": {"x1": 500, "y1": 100, "x2": 530, "y2": 160}},
                    {"category": "traffic sign", "box2d": {"x1": 700, "y1": 120, "x2": 730, "y2": 180}},
                    {"category": "car", "box2d": {"x1": 50, "y1": 400, "x2": 200, "y2": 650}},
                    {"category": "lane", "poly2d": [{"vertices": [{"x": 200, "y": 500}, {"x": 240, "y": 700}]}]},
                ]
            }
        ],
        "drivable_available": True,
    }
    names = _targeted_names(builder.build_from_records([record], device=torch.device("cpu")), builder)
    assert "traffic_light_visible" in names
    assert "traffic_sign_visible" in names
    assert "vehicle_left" in names
    assert "drivable_center" in names
    assert "traffic_light_green" not in names
    assert "stop_sign_present" not in names
    assert "parked_vehicle_left" not in names
    assert "front_vehicle_close" not in names or "front_vehicle_far" not in names
    assert "open_left_gap" not in names
    assert "open_right_gap" not in names
    assert "left_turn_region" not in names
    assert "merging_left_context" not in names
