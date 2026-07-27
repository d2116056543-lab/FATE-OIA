from __future__ import annotations

import torch

from fate_oia.datasets.bdd100k_task_aware_index import RAELGroundingRecord
from fate_oia.datasets.rael_dynamic_reliability import build_dynamic_reliability
from fate_oia.datasets.rael_grounding_targets import build_dynamic_grounding_batch


def _records() -> tuple[RAELGroundingRecord, RAELGroundingRecord]:
    rows = []
    for side, box in (
        ("left", (4.0, 4.0, 20.0, 20.0)),
        ("right", (44.0, 4.0, 60.0, 20.0)),
    ):
        rows.append(
            RAELGroundingRecord(
                detections=(
                    {
                        "id": "shared-object",
                        "category": "vehicle",
                        "box": box,
                        "sector": side,
                        "attributes": {},
                    },
                ),
                lanes=(),
                drivable=(),
                source_complete={
                    "detections": True,
                    "lanes": False,
                    "drivable": False,
                },
            )
        )
    return tuple(rows)


def _dynamic(cost_offset: float = 0.0):
    records = _records()
    slots = (
        (
            {"category": "vehicle", "box": (4.0 + cost_offset, 4.0, 20.0, 20.0), "sector": "left"},
            {"category": "other", "box": (30.0, 4.0, 40.0, 20.0), "sector": "center"},
        ),
        (
            {"category": "other", "box": (24.0, 4.0, 34.0, 20.0), "sector": "center"},
            {"category": "vehicle", "box": (44.0, 4.0, 60.0, 20.0), "sector": "right"},
        ),
    )
    return build_dynamic_grounding_batch(slots, records, ((64, 32), (64, 32)))


def _outputs() -> dict[str, object]:
    masks = torch.zeros(2, 20, 4, 8)
    masks[0, 0, 1:3, 0:2] = 1.0
    masks[1, 1, 1:3, 6:8] = 1.0
    type_probs = torch.zeros(2, 2, 6)
    type_probs[:, :, 0] = 1.0
    state_probs = torch.zeros(2, 2, 4)
    state_probs[:, :, 0] = 1.0
    return {
        "slot_masks": masks,
        "slot_observability": torch.full((2, 20), 0.8, requires_grad=True),
        "slot_type_probs": type_probs,
        "slot_state_probs": state_probs,
        "grounding_outputs": {
            "road": {"drivable_reliability": torch.full((2, 3), 0.75)}
        },
    }


def test_feature_dropout_bootstraps_only_grounded_observable_slots_without_view_history() -> None:
    outputs = _outputs()
    outputs["slot_observability"] = torch.full((2, 20), 0.8, requires_grad=True)
    consistency = torch.zeros(2, 20)
    consistency[0, 0] = 0.65
    consistency[1, 1] = 0.65
    outputs["slot_feature_dropout_consistency"] = consistency
    result = build_dynamic_reliability(
        outputs,
        _dynamic(),
        _records(),
        _road_valid(),
        mirror_pairs=torch.empty(0, 2, dtype=torch.long),
        sample_ids=("cold-left", "cold-right"),
        ema_state={},
    )

    assert result.q_view[0, 0] == 0.65
    assert result.q_view[1, 1] == 0.65
    assert result.rho[0, 0] > 0.0
    assert result.rho[1, 1] > 0.0
    assert result.q_view_source[0, 0] == 3
    assert result.q_view_source[1, 1] == 3
    assert result.q_view_bootstrap_count == 2
    assert result.rho_nonzero_rate > 0.0
    assert result.q_view[0, 2] == 0.0


def test_view_history_has_priority_over_feature_dropout_bootstrap() -> None:
    outputs = _outputs()
    outputs["slot_feature_dropout_consistency"] = torch.full((2, 20), 0.2)
    result = build_dynamic_reliability(
        outputs,
        _dynamic(),
        _records(),
        _road_valid(),
        mirror_pairs=torch.tensor([[0, 1]]),
        sample_ids=("case", "case"),
        ema_state={},
    )

    assert result.q_view[0, 0] > 0.9
    assert result.q_view[1, 1] > 0.9
    assert result.q_view_source[0, 0] == 2
    assert result.q_view_source[1, 1] == 2


def _road_valid() -> dict[str, torch.Tensor]:
    return {
        "drivable_valid_mask": torch.zeros(2, 3, dtype=torch.bool),
        "boundary_valid_mask": torch.zeros(2, 2, dtype=torch.bool),
    }


def test_qground_is_cost_monotonic_and_unknown_is_zero() -> None:
    close = build_dynamic_reliability(
        _outputs(), _dynamic(0.0), _records(), _road_valid(),
        mirror_pairs=torch.tensor([[0, 1]]), sample_ids=("a", "a"), ema_state={}
    )
    far = build_dynamic_reliability(
        _outputs(), _dynamic(12.0), _records(), _road_valid(),
        mirror_pairs=torch.tensor([[0, 1]]), sample_ids=("a", "a"), ema_state={}
    )
    assert close.q_ground[0, 0] > far.q_ground[0, 0] > 0
    assert close.q_ground[0, 1] == 1.0


def test_qview_aligns_by_object_identity_not_slot_index_and_updates_ema() -> None:
    result = build_dynamic_reliability(
        _outputs(), _dynamic(), _records(), _road_valid(),
        mirror_pairs=torch.tensor([[0, 1]]), sample_ids=("case", "case"), ema_state={}
    )
    assert result.q_view[0, 0] > 0.9
    assert result.q_view[1, 1] > 0.9
    assert result.q_view[0, 1] == 0.0
    assert "shared-object" in result.ema_state["objects"]


def test_no_current_observation_without_ema_is_zero_and_ema_reuses_exactly() -> None:
    first = build_dynamic_reliability(
        _outputs(), _dynamic(), _records(), _road_valid(),
        mirror_pairs=torch.tensor([[0, 1]]), sample_ids=("case", "case"), ema_state={}
    )
    second = build_dynamic_reliability(
        _outputs(), _dynamic(), _records(), _road_valid(),
        mirror_pairs=torch.empty(0, 2, dtype=torch.long),
        sample_ids=("case", "case"), ema_state=first.ema_state
    )
    assert second.q_view[0, 0] == first.ema_state["objects"]["shared-object"]
    assert second.q_view[0, 1] == 0.0
    assert torch.count_nonzero(second.q_view[:, 17:20]).item() == 0


def test_reliability_formula_is_detached_and_state_confidence_is_applied() -> None:
    outputs = _outputs()
    result = build_dynamic_reliability(
        outputs, _dynamic(), _records(), _road_valid(),
        mirror_pairs=torch.tensor([[0, 1]]), sample_ids=("case", "case"), ema_state={}
    )
    expected = (
        outputs["slot_observability"].detach()
        * result.q_ground
        * result.q_view
        * result.q_state
    )
    torch.testing.assert_close(result.rho, expected)
    assert not result.rho.requires_grad
    assert not result.q_ground.requires_grad
    assert not result.q_view.requires_grad


def test_resume_ema_next_batch_is_exact_and_serializable() -> None:
    first = build_dynamic_reliability(
        _outputs(), _dynamic(), _records(), _road_valid(),
        mirror_pairs=torch.tensor([[0, 1]]), sample_ids=("case", "case"), ema_state={}
    )
    left = build_dynamic_reliability(
        _outputs(), _dynamic(), _records(), _road_valid(),
        mirror_pairs=torch.empty(0, 2, dtype=torch.long),
        sample_ids=("case", "case"), ema_state=first.ema_state
    )
    right = build_dynamic_reliability(
        _outputs(), _dynamic(), _records(), _road_valid(),
        mirror_pairs=torch.empty(0, 2, dtype=torch.long),
        sample_ids=("case", "case"), ema_state=dict(first.ema_state)
    )
    torch.testing.assert_close(left.q_view, right.q_view)
    assert left.ema_state == right.ema_state
