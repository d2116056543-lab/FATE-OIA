from __future__ import annotations

import math

import pytest
import torch

from fate_oia.datasets.bdd100k_task_aware_index import RAELGroundingRecord
from fate_oia.datasets.rael_grounding_targets import (
    build_dynamic_grounding_batch,
    road_grounding_tensor_targets,
    slot_descriptors_from_predictions,
)


def _record(
    *,
    box: tuple[float, float, float, float] = (8.0, 8.0, 24.0, 24.0),
    complete: bool = True,
) -> RAELGroundingRecord:
    return RAELGroundingRecord(
        detections=(
            {
                "category": "vehicle",
                "box": box,
                "sector": "left",
                "attributes": {},
            },
        ),
        lanes=(),
        drivable=(),
        source_complete={
            "detections": complete,
            "lanes": False,
            "drivable": False,
        },
    )


def _slots(
    first_box: tuple[float, float, float, float] = (8.0, 8.0, 24.0, 24.0),
) -> tuple[dict[str, object], ...]:
    return (
        {"category": "vehicle", "box": first_box, "sector": "left"},
        {"category": "pedestrian", "box": (40.0, 8.0, 52.0, 28.0), "sector": "right"},
    )


def test_dynamic_grounding_builds_each_sample_from_current_slot_descriptors() -> None:
    batch = build_dynamic_grounding_batch(
        (_slots(), _slots((36.0, 8.0, 52.0, 24.0))),
        (_record(), _record(box=(36.0, 8.0, 52.0, 24.0))),
        ((64, 32), (64, 32)),
    )

    assert len(batch.entity) == len(batch.road) == 2
    assert batch.entity[0].assignments[0].slot_index == 0
    assert batch.entity[1].assignments[0].slot_index == 0
    assert all(math.isfinite(item.cost) for target in batch.entity for item in target.assignments)
    assert batch.coverage["matched_entity_count"] == 2


def test_dynamic_grounding_assignment_and_cost_change_with_current_forward_slots() -> None:
    record = _record()
    aligned = build_dynamic_grounding_batch((_slots(),), (record,), ((64, 32),))
    swapped = build_dynamic_grounding_batch(
        (
            (
                {"category": "pedestrian", "box": (40.0, 8.0, 52.0, 28.0), "sector": "right"},
                {"category": "vehicle", "box": (8.0, 8.0, 24.0, 24.0), "sector": "left"},
            ),
        ),
        (record,),
        ((64, 32),),
    )

    assert aligned.entity[0].assignments[0].slot_index == 0
    assert swapped.entity[0].assignments[0].slot_index == 1
    assert aligned.entity[0].assignments[0].cost == pytest.approx(
        swapped.entity[0].assignments[0].cost
    )


def test_slot_descriptors_are_current_detached_prediction_dependent_values() -> None:
    centroid = torch.tensor(
        [[[-0.75, -0.50], [0.75, -0.50]]], requires_grad=True
    )
    scale = torch.tensor([[0.25, 0.25]], requires_grad=True)
    type_probs = torch.zeros(1, 2, 6, requires_grad=True)
    horizontal = torch.zeros(1, 2, 3, requires_grad=True)
    with torch.no_grad():
        type_probs[0, 0, 0] = 1.0
        type_probs[0, 1, 1] = 1.0
        horizontal[0, 0, 0] = 1.0
        horizontal[0, 1, 2] = 1.0

    descriptors = slot_descriptors_from_predictions(
        centroid,
        scale,
        type_probs,
        horizontal,
        ((64, 32),),
    )

    assert descriptors[0][0]["category"] == "vehicle"
    assert descriptors[0][0]["sector"] == "left"
    assert descriptors[0][1]["category"] == "pedestrian"
    assert descriptors[0][1]["sector"] == "right"
    assert descriptors[0][0]["box"] != descriptors[0][1]["box"]
    assert all(
        isinstance(value, float)
        for descriptor in descriptors[0]
        for value in descriptor["box"]
    )


def test_road_tensor_targets_rasterize_actual_transformed_geometry() -> None:
    record = RAELGroundingRecord(
        detections=(),
        lanes=(
            {
                "points": ((4.0, 2.0), (4.0, 30.0)),
                "side": "left",
                "attributes": {"lineStyle": "solid"},
            },
        ),
        drivable=(
            {
                "polygon": ((0.0, 16.0), (20.0, 16.0), (20.0, 31.0), (0.0, 31.0)),
                "side": "left",
            },
        ),
        source_complete={"detections": False, "lanes": True, "drivable": True},
    )
    dynamic = build_dynamic_grounding_batch((_slots(),), (record,), ((64, 32),))
    tensors = road_grounding_tensor_targets(
        dynamic.road[0],
        image_size=(64, 32),
        output_size=(8, 16),
        device="cpu",
    )

    assert tensors["drivable_targets"].shape == (1, 3, 8, 16)
    assert tensors["boundary_targets"].shape == (1, 2, 8, 16)
    assert tensors["drivable_valid_mask"].tolist() == [[True, False, False]]
    assert tensors["boundary_valid_mask"].tolist() == [[True, False]]
    assert float(tensors["drivable_targets"][0, 0].sum()) > 0.0
    assert float(tensors["boundary_targets"][0, 0].sum()) > 0.0
    assert float(tensors["drivable_targets"][0, 1:].sum()) == 0.0


def test_dynamic_grounding_preserves_incomplete_source_as_unknown() -> None:
    batch = build_dynamic_grounding_batch(
        (_slots(),),
        (_record(complete=False),),
        ((64, 32),),
    )

    unmatched = batch.entity[0].objectness[1]
    assert unmatched.reliable is False
    assert batch.coverage["unknown_count"] == 1
    assert batch.coverage["reliable_negative_count"] == 0


@pytest.mark.parametrize(
    ("slots", "records", "sizes"),
    [
        ((_slots(),), (), ((64, 32),)),
        ((_slots(),), (_record(),), ()),
        ((_slots(), _slots()), (_record(),), ((64, 32),)),
    ],
)
def test_dynamic_grounding_rejects_batch_length_mismatch(
    slots: tuple[tuple[dict[str, object], ...], ...],
    records: tuple[RAELGroundingRecord, ...],
    sizes: tuple[tuple[int, int], ...],
) -> None:
    with pytest.raises(ValueError, match="same non-zero batch length"):
        build_dynamic_grounding_batch(slots, records, sizes)


@pytest.mark.parametrize("image_size", [(0, 32), (64, 0), (-1, 32)])
def test_dynamic_grounding_rejects_invalid_image_size(
    image_size: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        build_dynamic_grounding_batch((_slots(),), (_record(),), (image_size,))
