from fate_oia.models.acpr_predicate_targets import WeakPredicateTargetBuilder


def test_strict_predicate_targets_do_not_infer_color_or_stop_or_parked():
    b = WeakPredicateTargetBuilder("configs/acpr_scene_predicates.yaml")
    rec = {
        "frames": [
            {
                "labels": [
                    {"category": "traffic light", "box2d": {"x1": 500, "x2": 520, "y1": 80, "y2": 130}},
                    {"category": "traffic sign", "box2d": {"x1": 600, "x2": 620, "y1": 100, "y2": 150}},
                    {"category": "car", "box2d": {"x1": 10, "x2": 200, "y1": 400, "y2": 650}},
                ]
            }
        ]
    }
    out = b.build_from_records([rec])
    names = b.names
    active = {names[i] for i, v in enumerate(out["predicate_targets"][0].tolist()) if v > 0.5}
    assert "traffic_light_visible" in active
    assert "traffic_light_green" not in active
    assert "stop_sign_present" not in active
    assert "parked_vehicle_left" not in active
    assert not ({"front_vehicle_close", "front_vehicle_far"} <= active)
