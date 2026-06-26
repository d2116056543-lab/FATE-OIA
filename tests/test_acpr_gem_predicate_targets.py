import torch

from fate_oia.models.acpr_predicate_targets import WeakPredicateTargetBuilder


def _value(batch, name, builder):
    return batch["predicate_targets"][0, builder.name_to_id[name]].item(), batch["predicate_mask"][0, builder.name_to_id[name]].item()


def test_weak_predicate_cleanup_uses_unknown_mask_not_false_positive():
    builder = WeakPredicateTargetBuilder("configs/acpr_scene_predicates.yaml")
    rec = {"frames": [{"objects": [
        {"category": "traffic light", "box2d": {"x1": 100, "y1": 50, "x2": 140, "y2": 100}},
        {"category": "traffic sign", "box2d": {"x1": 200, "y1": 50, "x2": 240, "y2": 100}},
        {"category": "car", "box2d": {"x1": 500, "y1": 420, "x2": 760, "y2": 650}},
    ]}]}
    batch = builder.build_from_records([rec], device=torch.device("cpu"))

    assert _value(batch, "traffic_light_visible", builder) == (1.0, 1.0)
    assert _value(batch, "traffic_light_green", builder) == (0.0, 0.0)
    assert _value(batch, "stop_sign_present", builder) == (0.0, 0.0)
    close = _value(batch, "front_vehicle_close", builder)
    far = _value(batch, "front_vehicle_far", builder)
    assert close[0] + far[0] <= 1.0
    assert _value(batch, "road_clear", builder) == (0.0, 0.0)
