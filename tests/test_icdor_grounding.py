from __future__ import annotations

from pathlib import Path

import torch
import yaml
from PIL import Image

from fate_oia.datasets.mosaic_icdor_grounding import ICDORGroundingObservationBuilder


def test_grounding_marks_only_available_geometry_and_keeps_missing_sources_unknown() -> None:
    factors = [
        {"name": "traffic_light_visible", "type": "point", "grounding_sources": ["box2d"]},
        {"name": "left_lane_marking_visible", "type": "curve", "grounding_sources": ["lane_polyline"]},
        {"name": "center_drivable_region_visible", "type": "region", "grounding_sources": ["drivable_mask"]},
    ]
    builder = ICDORGroundingObservationBuilder(factors, grid_hw=(4, 8))
    result = builder([
        {
            "image_size": [100, 200],
            "objects": [{"category": "traffic light", "box2d": {"x1": 70, "y1": 10, "x2": 100, "y2": 40}}],
            "lanes": [],
            "drivable_mask": torch.ones(4, 8),
        },
        {"image_size": [100, 200]},
    ], device=torch.device("cpu"), split="train")

    assert result["presence_target"].shape == (2, 3)
    assert result["presence_known_mask"][0, 0] == 1
    assert result["presence_target"][0, 0] == 1
    assert result["geometry_known_mask"][0, 0] == 1
    assert result["presence_known_mask"][1].sum() == 0
    assert result["weak_negative_mask"][0, 1] == 1
    assert result["presence_known_mask"][0, 1] == 0


def test_grounding_honors_declared_negative_policy_and_keeps_unidentifiable_cues_unknown() -> None:
    factors = [
        {
            "name": "traffic_light_visible",
            "type": "point",
            "grounding_sources": ["box2d"],
            "negative_policy": "reliable_if_source_complete",
        },
        {
            "name": "front_vehicle_risk_proxy",
            "type": "object",
            "grounding_sources": ["box2d"],
            "negative_policy": "unknown_without_depth",
        },
    ]
    result = ICDORGroundingObservationBuilder(factors, grid_hw=(4, 8))(
        [{"image_size": [100, 200], "objects": []}],
        device=torch.device("cpu"),
        split="train",
    )

    # A complete object source can certify that an observable traffic light is
    # absent. A depth-free risk proxy must remain unobserved instead.
    assert result["presence_known_mask"][0].tolist() == [1.0, 0.0]
    assert result["weak_negative_mask"][0].tolist() == [0.0, 0.0]


def test_grounding_matches_colored_traffic_lights_and_requires_complete_attributes_for_negative() -> None:
    factors = [
        {
            "name": "red_light_visible",
            "type": "point",
            "grounding_sources": ["box2d", "bdd100k_attributes"],
            "attribute_constraints": {"trafficLightColor": "red"},
            "negative_policy": "reliable_if_attribute_complete",
        },
        {
            "name": "green_light_visible",
            "type": "point",
            "grounding_sources": ["box2d", "bdd100k_attributes"],
            "attribute_constraints": {"trafficLightColor": "green"},
            "negative_policy": "reliable_if_attribute_complete",
        },
    ]
    builder = ICDORGroundingObservationBuilder(factors, grid_hw=(4, 8))
    red = {
        "image_size": [100, 200],
        "objects": [{
            "category": "traffic light",
            "box2d": {"x1": 70, "y1": 10, "x2": 100, "y2": 40},
            "attributes": {"trafficLightColor": "red"},
        }],
    }
    incomplete = {
        "image_size": [100, 200],
        "objects": [{
            "category": "traffic light",
            "box2d": {"x1": 70, "y1": 10, "x2": 100, "y2": 40},
            "attributes": {},
        }],
    }

    red_result = builder([red], device=torch.device("cpu"), split="train")
    assert red_result["presence_target"][0].tolist() == [1.0, 0.0]
    assert red_result["presence_known_mask"][0].tolist() == [1.0, 1.0]

    incomplete_result = builder([incomplete], device=torch.device("cpu"), split="train")
    assert incomplete_result["presence_known_mask"][0].tolist() == [0.0, 0.0]
    assert incomplete_result["weak_negative_mask"][0].tolist() == [1.0, 1.0]


def test_grounding_rejects_test_split_and_reason_labels_are_not_an_input() -> None:
    builder = ICDORGroundingObservationBuilder([{"name": "traffic_light_visible", "type": "point", "grounding_sources": ["box2d"]}])
    try:
        builder([{}], device=torch.device("cpu"), split="test")
    except ValueError as error:
        assert "train-only" in str(error)
    else:
        raise AssertionError("test geometry must not enter IC-DOR grounding observation")


def test_grounding_keeps_left_right_and_center_geometry_in_their_declared_corridors() -> None:
    factors = [
        {"name": "left_obstacle_visible", "type": "object", "grounding_sources": ["box2d"], "weak_regions": ["left_corridor"]},
        {"name": "right_obstacle_visible", "type": "object", "grounding_sources": ["box2d"], "weak_regions": ["right_corridor"]},
        {"name": "left_lane_marking_visible", "type": "curve", "grounding_sources": ["lane_polyline"], "weak_regions": ["left_corridor"]},
        {"name": "right_lane_marking_visible", "type": "curve", "grounding_sources": ["lane_polyline"], "weak_regions": ["right_corridor"]},
        {"name": "left_drivable_region_visible", "type": "region", "grounding_sources": ["drivable_mask"], "weak_regions": ["left_corridor"]},
        {"name": "right_drivable_region_visible", "type": "region", "grounding_sources": ["drivable_mask"], "weak_regions": ["right_corridor"]},
        {"name": "center_drivable_region_visible", "type": "region", "grounding_sources": ["drivable_mask"], "weak_regions": ["center_corridor"]},
    ]
    builder = ICDORGroundingObservationBuilder(factors, grid_hw=(4, 8))
    right_only = torch.zeros(4, 8)
    right_only[:, 6:] = 1
    result = builder([
        {
            "image_size": [100, 200],
            "objects": [{"category": "car", "box2d": {"x1": 150, "y1": 55, "x2": 190, "y2": 95}}],
            "lanes": [{"category": "lane", "poly2d": [{"vertices": [[170, 20], [175, 90]]}]}],
            "drivable_mask": right_only,
        },
    ], device=torch.device("cpu"), split="train")

    # A right-side observation must not certify left or center factors.
    assert result["presence_target"][0].tolist() == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert result["weak_negative_mask"][0].tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert result["geometry_masks"][0, 1, :, :4].sum() == 0
    assert result["geometry_masks"][0, 5, :, :4].sum() == 0


def test_unidentifiable_cues_are_latent_or_image_only_and_never_action_edges() -> None:
    config_root = Path(__file__).resolve().parents[1] / "configs"
    with (config_root / "mosaic_icdor_factor_candidates.yaml").open("r", encoding="utf-8") as handle:
        factors = {entry["name"]: entry for entry in yaml.safe_load(handle)["factors"]}
    with (config_root / "mosaic_icdor_action_routes.yaml").open("r", encoding="utf-8") as handle:
        action_routes = yaml.safe_load(handle)["action_routes"]

    proxy = factors["front_vehicle_risk_proxy"]
    assert proxy["role"] == "latent_only"
    assert proxy["source_kind"] == "proxy"
    assert proxy["grounding_sources"] == ["box2d"]

    prohibited = {
        "front_vehicle_risk_proxy",
        "front_vehicle_left_indicator_visible",
        "front_vehicle_right_indicator_visible",
    }
    assert all(factors[name]["grounding_sources"] == ["image_only"] for name in prohibited - {"front_vehicle_risk_proxy"})
    route_factors = {
        edge["factor"]
        for route in action_routes.values()
        for edge in [*route.get("support", []), *route.get("veto", [])]
    }
    assert prohibited.isdisjoint(route_factors)


def test_grounding_loads_real_drivable_map_paths(tmp_path: Path) -> None:
    path = tmp_path / "sample_drivable_id.png"
    image = Image.new("L", (8, 4), color=0)
    for y in range(2, 4):
        for x in range(5, 8):
            image.putpixel((x, y), 1)
    image.save(path)
    factors = [{
        "name": "right_drivable_region_visible",
        "type": "region",
        "grounding_sources": ["drivable_mask"],
        "weak_regions": ["right_corridor"],
    }]
    result = ICDORGroundingObservationBuilder(factors, grid_hw=(4, 8))(
        [{"image_size": [4, 8], "drivable_map": str(path)}],
        device=torch.device("cpu"),
        split="train",
    )
    assert result["source_available"][0, 0]
    assert result["presence_target"][0, 0] == 1
    assert result["geometry_masks"][0, 0].sum() > 0
