from pathlib import Path

from fate_oia.datasets.meter_signed_targets import METERSignedTargetBuilder


def test_missing_attributes_remain_unknown() -> None:
    builder = METERSignedTargetBuilder(Path("configs/meter_factor_schema.yaml"), grid_hw=(8, 12))
    result = builder.build({"objects": [{"category": "traffic light", "box2d": {"x1": 10, "y1": 10, "x2": 30, "y2": 30}}]})
    assert not bool(result["factor_support_valid"][0])
    assert bool(result["factor_support_valid"][3])
    assert not bool(result["factor_support_valid"][5])


def test_lane_turn_semantics_are_explicit() -> None:
    builder = METERSignedTargetBuilder(Path("configs/meter_factor_schema.yaml"), grid_hw=(8, 12))
    result = builder.build({"lanes": [{"polyline": [[10, 100], [20, 200]], "turn": "right"}]})
    assert not bool(result["factor_support_valid"][12])
    assert bool(result["factor_support_valid"][18])
