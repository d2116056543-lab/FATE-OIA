from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_NAMES = [
    "Traffic light is green", "Follow traffic", "Road is clear", "Traffic light", "Traffic sign",
    "Obstacle: car", "Obstacle: person", "Obstacle: rider", "Obstacle: others", "No lane on the left",
    "Obstacles on the left lane", "Solid line on the left", "On the left-turn lane", "Traffic light allows left",
    "Front car turning left", "No lane on the right", "Obstacles on the right lane", "Solid line on the right",
    "On the right-turn lane", "Traffic light allows right", "Front car turning right",
]


def test_factor_schema_is_complete_and_provenanced() -> None:
    path = ROOT / "configs" / "meter_factor_schema.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["reason_index_mapping"]["source"]
    assert data["reason_index_mapping"]["version"]
    assert data["reason_index_mapping"]["human_confirmed"] is False

    factors = data["factors"]
    assert len(factors) == 21
    assert [item["id"] for item in factors] == list(range(21))
    assert [item["name"] for item in factors] == EXPECTED_NAMES
    for item in factors:
        for key in (
            "group", "region", "mirror_partner", "support_predicates", "counter_predicates",
            "groundability", "tail", "pu_eligible",
        ):
            assert key in item
        assert item["groundability"] in {"full", "partial", "latent"}


def test_grounding_schema_requires_unknown_for_ambiguous_observations() -> None:
    path = ROOT / "configs" / "meter_grounding_schema.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules = data["conservative_rules"]
    assert rules["unknown_traffic_light_color"] == "unknown"
    assert rules["generic_traffic_sign"] == "unknown"
    assert rules["missing_lane_style"] == "unknown"
    assert rules["missing_object_annotation"] == "unknown"
    assert rules["missing_drivable_map"] == "unknown"
    assert rules["static_box_turn_intent"] == "unknown"
