from __future__ import annotations

import pytest
import torch

from fate_oia.datasets.mosaic_grounding_observations import MOSAICGroundingObservationBuilder


FACTORS = (
    {
        "name": "front_vehicle_visible",
        "type": "object",
        "entity": "vehicle",
        "attribute": None,
        "spatial": "front_center",
        "reason_positive_anchors": [5],
        "geometry_sources": ["box2d"],
    },
    {
        "name": "front_vehicle_near",
        "type": "object",
        "entity": "vehicle",
        "attribute": "near",
        "spatial": "front_center",
        "reason_positive_anchors": [5],
        "geometry_sources": ["box2d"],
    },
    {
        "name": "left_lane_marking_visible",
        "type": "curve",
        "entity": "lane_marking",
        "attribute": None,
        "spatial": "left_corridor",
        "reason_positive_anchors": [],
        "geometry_sources": ["lane_polyline"],
    },
    {
        "name": "center_drivable_visible",
        "type": "region",
        "entity": "drivable",
        "attribute": None,
        "spatial": "center_corridor",
        "reason_positive_anchors": [2],
        "geometry_sources": ["drivable_mask"],
    },
)


def _reasons(batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, 21)


def test_positive_reason_creates_only_permitted_weak_positive_observations() -> None:
    builder = MOSAICGroundingObservationBuilder(FACTORS)
    reasons = _reasons()
    reasons[0, 5] = 1

    output = builder(reasons, [None], split="train")

    assert output["presence_target"].tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert output["presence_mask"].tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert output["visibility_target"].tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert output["visibility_mask"].tolist() == [[1.0, 0.0, 0.0, 0.0]]
    assert output["source_code"].tolist() == [[1, 0, 0, 0]]
    assert torch.count_nonzero(output["geometry_mask_valid"]) == 0


def test_reason_zero_and_missing_annotations_remain_unknown_not_negative() -> None:
    builder = MOSAICGroundingObservationBuilder(FACTORS)
    output = builder(_reasons(), [None], split="train")

    for key in ("presence_target", "presence_mask", "visibility_target", "visibility_mask"):
        assert torch.count_nonzero(output[key]) == 0
    assert torch.count_nonzero(output["source_reliability"]) == 0
    assert torch.count_nonzero(output["source_code"]) == 0
    assert torch.count_nonzero(output["geometry_mask"]) == 0
    assert torch.count_nonzero(output["geometry_mask_valid"]) == 0


def test_box_lane_and_drivable_sources_supervise_only_compatible_factors() -> None:
    builder = MOSAICGroundingObservationBuilder(FACTORS)
    drivable = torch.zeros(100, 200)
    drivable[45:, 75:125] = 1
    record = {
        "image_size": (100, 200),
        "objects": [{"category": "car", "box2d": {"x1": 80, "y1": 35, "x2": 120, "y2": 90}}],
        "lanes": [{"category": "lane marking", "poly2d": [{"vertices": [[20, 40], [45, 95]]}]}],
        "drivable_mask": drivable,
    }

    output = builder(_reasons(), [record], split="train")

    assert output["presence_mask"].tolist() == [[1.0, 0.0, 1.0, 1.0]]
    assert output["presence_target"].tolist() == [[1.0, 0.0, 1.0, 1.0]]
    assert output["source_code"].tolist() == [[2, 0, 3, 4]]
    assert output["geometry_mask_valid"].tolist() == [[1.0, 0.0, 1.0, 1.0]]
    assert torch.all(output["geometry_mask"].sum(dim=(-2, -1))[0, [0, 2, 3]] > 0)
    assert output["source_reliability"][0, 3] > output["source_reliability"][0, 2]


def test_explicit_attribute_is_required_for_near_or_directional_box_factors() -> None:
    builder = MOSAICGroundingObservationBuilder(FACTORS)
    base = {"category": "car", "box2d": {"x1": 80, "y1": 35, "x2": 120, "y2": 90}}
    no_attribute = builder(_reasons(), [{"image_size": (100, 200), "objects": [base]}], split="train")
    with_attribute = builder(
        _reasons(),
        [{"image_size": (100, 200), "objects": [{**base, "attributes": {"distance": "near"}}]}],
        split="train",
    )
    assert no_attribute["presence_mask"][0, 1] == 0
    assert with_attribute["presence_mask"][0, 1] == 1


def test_complete_geometry_sources_emit_weak_negative_presence_without_treating_unknown_attributes_as_negative() -> None:
    builder = MOSAICGroundingObservationBuilder(FACTORS)
    record = {
        "image_size": (100, 200),
        "objects": [],
        "lanes": [],
        "drivable_mask": torch.zeros(100, 200),
    }

    output = builder(_reasons(), [record], split="train")

    # Source-complete, attribute-free factors have an observed absence. The
    # model can inspect the region (visibility=1) while presence is negative.
    assert output["presence_target"].tolist() == [[0.0, 0.0, 0.0, 0.0]]
    assert output["presence_mask"].tolist() == [[1.0, 0.0, 1.0, 1.0]]
    assert output["visibility_target"].tolist() == [[1.0, 0.0, 1.0, 1.0]]
    assert output["visibility_mask"].tolist() == [[1.0, 0.0, 1.0, 1.0]]
    assert output["weak_negative_mask"].tolist() == [[1.0, 0.0, 1.0, 1.0]]
    assert output["source_code"].tolist() == [[2, 0, 3, 4]]
    assert torch.all(output["source_reliability"][0, [0, 2, 3]] > 0)
    assert torch.all(output["source_reliability"][0, [0, 2, 3]] < 0.5)


def test_reason_anchor_does_not_override_a_complete_geometry_negative() -> None:
    builder = MOSAICGroundingObservationBuilder(FACTORS)
    reasons = _reasons()
    reasons[0, 5] = 1
    record = {"image_size": (100, 200), "objects": [], "lanes": []}

    output = builder(reasons, [record], split="train")

    assert output["presence_target"][0, 0] == 0
    assert output["presence_mask"][0, 0] == 1
    assert output["weak_negative_mask"][0, 0] == 1
    # `near` is not directly observable from the current BDD100K schema, so
    # its missing attribute remains unknown rather than becoming a hard zero.
    assert output["presence_mask"][0, 1] == 0


def test_occupancy_ignores_traffic_control_boxes_and_uses_exclusive_corridors() -> None:
    occupancy_factors = tuple(
        {
            "name": f"{spatial}_occupied",
            "type": "region",
            "entity": "occupancy",
            "attribute": None,
            "spatial": spatial,
            "reason_positive_anchors": [],
            "geometry_sources": ["box2d"],
        }
        for spatial in ("left_corridor", "center_corridor", "right_corridor")
    )
    builder = MOSAICGroundingObservationBuilder(occupancy_factors)
    record = {
        "image_size": (100, 200),
        "objects": [
            {"category": "traffic light", "box2d": {"x1": 15, "y1": 30, "x2": 35, "y2": 60}},
            {"category": "car", "box2d": {"x1": 85, "y1": 45, "x2": 115, "y2": 95}},
        ],
    }

    output = builder(_reasons(), [record], split="train")

    # The center car must not simultaneously become left/right occupancy,
    # and the traffic light is not an obstacle occupying the driving lane.
    assert output["presence_target"].tolist() == [[0.0, 1.0, 0.0]]
    assert output["presence_mask"].tolist() == [[1.0, 1.0, 1.0]]
    assert output["weak_negative_mask"].tolist() == [[1.0, 0.0, 1.0]]


def test_builder_is_train_only_and_validates_batch_contract() -> None:
    builder = MOSAICGroundingObservationBuilder(FACTORS)
    with pytest.raises(ValueError, match="train-only"):
        builder(_reasons(), [None], split="test")
    with pytest.raises(ValueError, match=r"\[B,21\]"):
        builder(torch.zeros(1, 20), [None], split="train")
    with pytest.raises(ValueError, match="record count"):
        builder(_reasons(batch=2), [None], split="train")
