"""P6 RED/GREEN contracts for RAEL slot attributes and reliability."""

from __future__ import annotations

import importlib

import pytest
import torch


def _module():
    try:
        return importlib.import_module("fate_oia.losses.rael_grounding_losses")
    except ModuleNotFoundError as error:
        pytest.fail(f"P6 grounding module is not implemented: {error}")


def _inputs(batch: int = 2, dim: int = 8):
    torch.manual_seed(6)
    entity_tokens = torch.randn(batch, 12, dim)
    road_tokens = torch.randn(batch, 5, dim)
    entity_masks = torch.zeros(batch, 12, 4, 6)
    entity_masks[:, 0, :, :2] = 1.0
    entity_masks[:, 1, 2:, 4:] = 1.0
    road_masks = torch.zeros(batch, 5, 4, 6)
    road_masks[:, 0, :, :2] = 1.0
    road_masks[:, 1, :, 2:4] = 1.0
    road_masks[:, 2, :, 4:] = 1.0
    return entity_tokens, entity_masks, road_tokens, road_masks


def test_p6_attribute_head_has_exact_vocabularies_and_conditional_outputs() -> None:
    module = _module()
    head = module.RAELSlotAttributeHeads(dim=8)
    entity_tokens, entity_masks, road_tokens, road_masks = _inputs()
    output = head(entity_tokens, entity_masks, road_tokens, road_masks)

    assert module.ENTITY_TYPES == (
        "vehicle",
        "pedestrian",
        "rider",
        "traffic_control",
        "traffic_sign",
        "other",
    )
    assert module.TRAFFIC_STATES == ("red", "green", "yellow_or_other", "unknown")
    assert module.BOUNDARY_STYLES == ("solid", "dashed_or_other", "unknown")
    assert output["presence_logits"].shape == (2, 12)
    assert output["observability_logits"].shape == (2, 12)
    assert output["entity_type_logits"].shape == (2, 12, 6)
    assert output["traffic_state_logits"].shape == (2, 12, 4)
    assert output["drivable_logits"].shape == (2, 3, 4, 6)
    assert output["boundary_style_logits"].shape == (2, 2, 3)
    assert output["horizontal_sector_probs"].shape == (2, 12, 3)
    assert output["depth_sector_probs"].shape == (2, 12, 3)
    assert torch.allclose(output["entity_type_probs"].sum(dim=-1), torch.ones(2, 12), atol=1e-6)
    assert torch.allclose(output["traffic_state_probs"].sum(dim=-1), torch.ones(2, 12), atol=1e-6)
    assert torch.allclose(output["boundary_style_probs"].sum(dim=-1), torch.ones(2, 2), atol=1e-6)


def test_p6_sector_geometry_dominates_weak_mlp_and_wrong_token_only_double_fails() -> None:
    module = _module()
    head = module.RAELSlotAttributeHeads(dim=8, sector_aux_scale=0.05)
    with torch.no_grad():
        for parameter in (head.horizontal_sector_aux.weight, head.horizontal_sector_aux.bias):
            parameter.zero_()
    entity_tokens, entity_masks, road_tokens, road_masks = _inputs(batch=1)
    output = head(entity_tokens, entity_masks, road_tokens, road_masks)
    horizontal = output["horizontal_sector_probs"]
    assert horizontal[0, 0, 0] > horizontal[0, 0, 2]
    assert horizontal[0, 1, 2] > horizontal[0, 1, 0]

    token_only = torch.softmax(head.horizontal_sector_aux(entity_tokens), dim=-1)
    with pytest.raises(AssertionError):
        assert torch.allclose(token_only, horizontal, atol=1e-5)


def test_p6_depth_sectors_follow_driving_perspective_not_image_row_order() -> None:
    """The road-facing bottom of the image is near; the sky-facing top is far."""
    module = _module()
    head = module.RAELSlotAttributeHeads(dim=8, sector_aux_scale=0.0)
    entity_tokens, entity_masks, road_tokens, road_masks = _inputs(batch=1)
    entity_masks.zero_()
    entity_masks[:, 0, -1, :] = 1.0  # lower image: nearest driving region
    entity_masks[:, 1, entity_masks.shape[-2] // 2, :] = 1.0
    entity_masks[:, 2, 0, :] = 1.0  # upper image: furthest driving region

    output = head(entity_tokens, entity_masks, road_tokens, road_masks)
    depth = output["depth_sector_probs"]
    assert depth[0, 0, module.DEPTH_SECTORS.index("near")] > 0.99
    assert depth[0, 1, module.DEPTH_SECTORS.index("middle")] > 0.99
    assert depth[0, 2, module.DEPTH_SECTORS.index("far")] > 0.99


def test_p6_entity_reliability_uses_observability_not_presence_and_loss_detaches_weight() -> None:
    module = _module()
    observability = torch.tensor([[0.9, 0.4]], requires_grad=True)
    q_ground = torch.tensor([[0.8, 0.6]], requires_grad=True)
    q_view = torch.tensor([[0.7, 0.5]], requires_grad=True)
    q_state = torch.tensor([[0.6, 0.3]], requires_grad=True)
    reliability = module.entity_reliability(observability, q_ground, q_view, q_state)
    expected = observability * q_ground * q_view * q_state
    assert torch.allclose(reliability, expected)
    # Presence does not appear in rho, so a near-absent entity can still be a
    # reliably observed negative.
    low_presence = torch.tensor([[0.001, 0.999]])
    assert torch.allclose(reliability, expected)
    assert low_presence[0, 0] < 0.01

    logits = torch.tensor([[0.2, -0.1]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0]])
    valid = torch.tensor([[True, True]])
    result = module.reliability_weighted_bce(logits, targets, valid, reliability)
    result.loss.backward()
    assert observability.grad is None
    assert q_ground.grad is None
    assert q_view.grad is None
    assert q_state.grad is None
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_p6_type_and_traffic_state_masks_are_conditional_not_unknown_as_negative() -> None:
    module = _module()
    head = module.RAELSlotAttributeHeads(dim=8)
    entity_tokens, entity_masks, road_tokens, road_masks = _inputs(batch=1)
    output = head(entity_tokens, entity_masks, road_tokens, road_masks)
    traffic = module.ENTITY_TYPES.index("traffic_control")
    targets = {
        "presence": torch.tensor([[1.0] + [0.0] * 11]),
        "presence_valid": torch.tensor([[True] + [False] * 11]),
        # 999 is deliberately an out-of-range unknown sentinel.  Its false
        # mask must exclude it before cross entropy sees a class target.
        "type": torch.tensor([[traffic, 999] + [-1] * 10]),
        "type_valid": torch.tensor([[True, False] + [False] * 10]),
        "traffic_state": torch.tensor([[module.TRAFFIC_STATES.index("green"), -1] + [-1] * 10]),
        "traffic_state_valid": torch.tensor([[True, True] + [False] * 10]),
    }
    losses = module.entity_attribute_grounding_loss(output, targets)
    assert losses["presence"].active is True
    assert losses["entity_type"].active is True
    # Slot 1 requests a state but is an unknown non-traffic type, so it is
    # deliberately excluded rather than converted into a negative state label.
    assert losses["traffic_state"].valid_count == 1
    assert losses["traffic_state"].active is True


def test_p6_mirror_swaps_all_horizontal_sector_and_road_identities() -> None:
    module = _module()
    horizontal = torch.tensor([[[0.7, 0.2, 0.1]]])
    road = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
    mirrored = module.mirror_sector_and_road_ids(horizontal, road)
    assert torch.equal(mirrored["horizontal_sector_probs"], torch.tensor([[[0.1, 0.2, 0.7]]]))
    assert torch.equal(mirrored["road_values"], torch.tensor([[[3.0], [2.0], [1.0], [5.0], [4.0]]]))


def test_p6_mirror_spatial_road_maps_flips_pixels_before_fixed_identity_permutation() -> None:
    module = _module()
    horizontal = torch.tensor([[[0.7, 0.2, 0.1]]])
    spatial = torch.arange(5 * 45 * 80, dtype=torch.float32).reshape(1, 5, 45, 80)
    mirrored = module.mirror_sector_and_road_ids(horizontal, spatial)
    expected = spatial.flip(-1).index_select(
        1, torch.tensor(module.ROAD_MIRROR_PERMUTATION)
    )
    assert torch.equal(mirrored["road_values"], expected)


def test_p6_adapts_p1_hungarian_targets_without_inventing_unknown_attributes() -> None:
    module = _module()
    from fate_oia.datasets.rael_grounding_targets import (
        EntityAssignment,
        EntityGroundingTargets,
        ObjectnessTarget,
        TrafficStateTarget,
    )

    objectness = tuple(
        ObjectnessTarget(1.0 if index == 0 else 0.0, index in (0, 1), 0 if index == 0 else None)
        for index in range(12)
    )
    p1 = EntityGroundingTargets(
        assignments=(EntityAssignment(slot_index=0, detection_index=0, cost=0.25),),
        objectness=objectness,
        traffic_state_targets=(TrafficStateTarget(0, 0, "green", True),),
        coverage={},
    )
    converted = module.entity_attribute_targets_from_p1(
        p1,
        [{"category": "traffic_light"}],
        device="cpu",
    )
    traffic = module.ENTITY_TYPES.index("traffic_control")
    assert converted["type"][0, 0].item() == traffic
    assert converted["traffic_state"][0, 0].item() == module.TRAFFIC_STATES.index("green")
    assert converted["type_valid"][0, 1].item() is False
    assert converted["presence_valid"][0, 2].item() is False
    assert converted["q_ground"][0, 0].item() < 1.0


def test_p6_owner_api_exposes_real_unique_named_parameter_sets() -> None:
    module = _module()
    assert hasattr(module, "p6_parameter_ownership"), "P6 requires an auditable ownership API"
    head = module.RAELSlotAttributeHeads(dim=8)
    from fate_oia.models.rael_slot_ledger import RAELSlotLedger

    ledger_model = RAELSlotLedger(dim=8)
    report = module.p6_parameter_ownership(head, ledger_model)
    actual = dict(head.named_parameters())
    ledger_actual = dict(ledger_model.named_parameters())
    owned = report["slot_attribute_heads"]
    ledger = report["slot_ledger_core"]
    assert owned["owner"] == "slot_attribute_heads"
    assert set(owned["named_parameters"]) == set(actual)
    assert set(owned["parameter_ids"]) == {id(parameter) for parameter in actual.values()}
    assert len(owned["parameter_ids"]) == len(set(owned["parameter_ids"]))
    assert ledger["owner"] == "slot_ledger_core"
    assert report["verified"] is True
    assert set(ledger["named_parameters"]) == set(ledger_actual)
    assert set(ledger["parameter_ids"]) == {id(parameter) for parameter in ledger_actual.values()}
    assert not (set(owned["parameter_ids"]) & set(ledger["parameter_ids"]))
    assert set(ledger["readouts"]) == {"entity_masks", "road_masks", "road_slot_ids"}

    unverifiable = module.p6_parameter_ownership(head)
    assert unverifiable["verified"] is False
    assert unverifiable["slot_ledger_core"]["named_parameters"] == ()


def test_p6_boundary_style_respects_p1_assignment_and_masks_center_or_unknown() -> None:
    module = _module()
    strict_center = module.build_boundary_style_targets(
        [{"side": "left", "attributes": {"lineStyle": "solid"}}],
        assignments={0: "center"},
        device="cpu",
    )
    assert strict_center["valid_count"] == 0

    p1_numeric_right = module.build_boundary_style_targets(
        [{"side": "left", "attributes": {"lineStyle": "dashed"}}],
        assignments={0: {"sector": 2}},
        device="cpu",
    )
    assert p1_numeric_right["targets"].tolist() == [[-1, 1]]

    center_without_assignment = module.build_boundary_style_targets(
        [{"points": [(310.0, 100.0), (330.0, 200.0)], "attributes": {"lineStyle": "solid"}}],
        device="cpu",
        image_width=640.0,
    )
    assert center_without_assignment["valid_count"] == 0

    left_third = module.build_boundary_style_targets(
        [{"points": [(150.0, 100.0), (190.0, 200.0)], "attributes": {"lineStyle": "solid"}}],
        device="cpu",
        image_width=640.0,
    )
    right_third = module.build_boundary_style_targets(
        [{"points": [(470.0, 100.0), (520.0, 200.0)], "attributes": {"lineStyle": "dashed"}}],
        device="cpu",
        image_width=640.0,
    )
    assert left_third["targets"].tolist() == [[0, -1]]
    assert right_third["targets"].tolist() == [[-1, 1]]
