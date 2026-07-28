from pathlib import Path

from fate_oia.datasets.meter_signed_targets import METERSignedTargetBuilder


ROOT = Path(__file__).resolve().parents[1]


def test_conservative_builder_does_not_turn_unknown_observations_into_labels() -> None:
    builder = METERSignedTargetBuilder(ROOT / "configs" / "meter_factor_schema.yaml")
    target = builder.build({"source_complete": False, "objects": [{"category": "traffic light", "box2d": {"x1": 10, "y1": 10, "x2": 40, "y2": 40}}]})
    assert target["factor_support_valid"][0].item() is False
    assert target["factor_counter_valid"][0].item() is False
    assert target["factor_support_valid"][3].item() is True
    assert target["factor_support_valid"][4].item() is False


def test_green_light_and_explicit_center_car_create_separate_signed_targets() -> None:
    builder = METERSignedTargetBuilder(ROOT / "configs" / "meter_factor_schema.yaml")
    target = builder.build(
        {
            "source_complete": True,
            "objects": [
                {"category": "traffic light", "attributes": {"trafficLightColor": "green"}, "box2d": {"x1": 300, "y1": 20, "x2": 330, "y2": 80}},
                {"category": "car", "box2d": {"x1": 290, "y1": 220, "x2": 360, "y2": 330}},
            ],
        }
    )
    assert target["factor_support_valid"][0].item() is True
    assert target["factor_support_valid"][5].item() is True
    assert target["factor_support_map"][0].sum().item() > 0
    assert target["factor_support_map"][5].sum().item() > 0
