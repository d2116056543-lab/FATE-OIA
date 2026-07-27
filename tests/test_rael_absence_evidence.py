"""P6 RED/GREEN contracts for reliable absence and road grounding losses."""

from __future__ import annotations

import importlib

import pytest
import torch


def _module():
    try:
        return importlib.import_module("fate_oia.losses.rael_grounding_losses")
    except ModuleNotFoundError as error:
        pytest.fail(f"P6 grounding module is not implemented: {error}")


def test_p6_absence_exact_formula_and_clear_reliability() -> None:
    module = _module()
    presence = torch.tensor([[0.5, 0.25]])
    sectors = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    visibility = torch.tensor([[0.8, 0.7, 0.6]])
    q_view = torch.tensor([[0.9, 0.5, 0.4]], requires_grad=True)
    output = module.reliable_absence_evidence(presence, sectors, visibility, q_view)
    expected_occ = torch.tensor([[0.5, 0.0, 0.25]])
    assert torch.allclose(output["occupied"], expected_occ, atol=1e-6)
    assert torch.allclose(output["clear"], visibility * (1.0 - expected_occ), atol=1e-6)
    assert torch.allclose(output["clear_reliability"], visibility * q_view)
    assert output["clear_reliability"].requires_grad is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="P6 BF16 contract requires CUDA")
def test_p6_reliable_absence_is_cuda_bf16_autocast_safe_and_preserves_probability_semantics() -> None:
    """BF16-rounded softmax probabilities remain valid formal simplex inputs."""

    module = _module()
    device = torch.device("cuda")
    torch.manual_seed(17)
    presence_logits = (torch.randn(4, 12, device=device) * 2.0).requires_grad_()
    sector_logits = (torch.randn(4, 12, 3, device=device) * 4.0).requires_grad_()
    visibility_logits = torch.randn(4, 3, device=device, requires_grad=True)
    q_view_logits = torch.randn(4, 3, device=device, requires_grad=True)

    reference = module.reliable_absence_evidence(
        torch.sigmoid(presence_logits),
        torch.softmax(sector_logits, dim=-1),
        torch.sigmoid(visibility_logits),
        torch.sigmoid(q_view_logits),
    )

    presence_logits_bf16 = presence_logits.detach().to(torch.bfloat16).requires_grad_()
    sector_logits_bf16 = sector_logits.detach().to(torch.bfloat16).requires_grad_()
    visibility_logits_bf16 = visibility_logits.detach().to(torch.bfloat16).requires_grad_()
    q_view_logits_bf16 = q_view_logits.detach().to(torch.bfloat16).requires_grad_()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = module.reliable_absence_evidence(
            torch.sigmoid(presence_logits_bf16),
            torch.softmax(sector_logits_bf16, dim=-1),
            torch.sigmoid(visibility_logits_bf16),
            torch.sigmoid(q_view_logits_bf16),
        )
        loss = actual["occupied"].sum() + actual["clear"].sum()
    loss.backward()

    for name in ("occupied", "clear", "clear_reliability"):
        assert torch.isfinite(actual[name]).all()
        assert torch.allclose(actual[name].float(), reference[name].float(), atol=7.5e-3, rtol=7.5e-3)
    for gradient in (
        presence_logits_bf16.grad,
        sector_logits_bf16.grad,
        visibility_logits_bf16.grad,
    ):
        assert gradient is not None and torch.isfinite(gradient).all()
    # rho_clear intentionally detaches q_view so reliability cannot reduce its
    # own supervision weight through the absence-loss path.
    assert actual["clear_reliability"].requires_grad is False
    assert q_view_logits_bf16.grad is None


def test_p6_reliable_absence_still_rejects_non_simplex_sector_probabilities() -> None:
    module = _module()
    with pytest.raises(ValueError, match="sum to one"):
        module.reliable_absence_evidence(
            torch.full((1, 2), 0.5),
            torch.tensor([[[0.8, 0.2, 0.2], [0.8, 0.2, 0.2]]]),
            torch.full((1, 3), 0.7),
            torch.full((1, 3), 0.9),
        )


def test_p6_valid_count_zero_is_explicitly_inactive_not_a_fake_grounding_loss() -> None:
    module = _module()
    logits = torch.randn(2, 3, 4, 6, requires_grad=True)
    targets = torch.zeros_like(logits)
    valid = torch.zeros(2, 3, dtype=torch.bool)
    result = module.drivable_bce_dice_loss(logits, targets, valid)
    assert result.active is False
    assert result.valid_count == 0
    assert result.loss.requires_grad
    result.loss.backward()
    assert logits.grad is not None and torch.equal(logits.grad, torch.zeros_like(logits.grad))


def test_p6_drivable_bce_dice_and_boundary_losses_have_correct_direction() -> None:
    module = _module()
    target = torch.zeros(1, 3, 5, 7)
    target[:, :, 2, 3] = 1.0
    valid_road = torch.tensor([[True, False, True]])
    good = torch.where(target > 0, torch.full_like(target, 5.0), torch.full_like(target, -5.0)).requires_grad_()
    bad = -good.detach().clone().requires_grad_()
    good_loss = module.drivable_bce_dice_loss(good, target, valid_road).loss
    bad_loss = module.drivable_bce_dice_loss(bad, target, valid_road).loss
    assert good_loss < bad_loss

    boundary_target = target[:, :2]
    valid_boundary = torch.tensor([[True, True]])
    good_boundary = good[:, :2]
    bad_boundary = bad[:, :2]
    assert module.boundary_dilated_bce_cldice_symmetric_distance_loss(
        good_boundary, boundary_target, valid_boundary
    ).loss < module.boundary_dilated_bce_cldice_symmetric_distance_loss(
        bad_boundary, boundary_target, valid_boundary
    ).loss


def test_p6_boundary_unknown_masks_do_not_count_as_negatives_and_wrong_unmasked_double_fails() -> None:
    module = _module()
    logits = torch.zeros(1, 2, 3, 3)
    target = torch.zeros_like(logits)
    target[:, 0, 1, 1] = 1.0
    valid = torch.tensor([[True, False]])
    masked = module.boundary_dilated_bce_cldice_symmetric_distance_loss(logits, target, valid)
    assert masked.valid_count == 1 and masked.active is True

    wrong_unmasked = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    with pytest.raises(AssertionError):
        assert torch.allclose(masked.loss, wrong_unmasked, atol=1e-6)


def _line_logits(*, column: int, broken_row: int | None = None) -> torch.Tensor:
    logits = torch.full((1, 2, 9, 9), -8.0)
    logits[:, :, 1:8, column] = 8.0
    if broken_row is not None:
        logits[:, :, broken_row, column] = -8.0
    return logits


def test_p6_soft_skeleton_cldice_is_differentiable_and_penalizes_thin_breaks_and_offsets() -> None:
    module = _module()
    assert hasattr(module, "soft_skeletonize"), "P6 needs a real iterative soft skeleton operator"
    target = (_line_logits(column=4) > 0).to(torch.float32)
    valid = torch.tensor([[True, True]])
    continuous = _line_logits(column=4).requires_grad_()
    broken = _line_logits(column=4, broken_row=4)
    offset = _line_logits(column=5)
    continuous_loss = module.boundary_dilated_bce_cldice_symmetric_distance_loss(
        continuous, target, valid
    )
    broken_loss = module.boundary_dilated_bce_cldice_symmetric_distance_loss(broken, target, valid)
    offset_loss = module.boundary_dilated_bce_cldice_symmetric_distance_loss(offset, target, valid)
    assert {"topology_precision", "topology_sensitivity", "cldice"} <= set(continuous_loss.components)
    assert continuous_loss.components["cldice"] < broken_loss.components["cldice"]
    assert continuous_loss.components["cldice"] < offset_loss.components["cldice"]
    continuous_loss.loss.backward()
    assert continuous.grad is not None and torch.isfinite(continuous.grad).all()

    skeleton = module.soft_skeletonize(torch.sigmoid(continuous.detach()))
    assert skeleton.shape == continuous.shape
    assert torch.isfinite(skeleton).all()


def test_p6_symmetric_distance_has_both_true_directions_and_one_way_mutation_fails() -> None:
    module = _module()
    assert hasattr(module, "symmetric_distance_transform_loss"), "P6 needs two-way differentiable distance transforms"
    target = torch.zeros(1, 2, 9, 9)
    target[:, :, 1:8, 2] = 1.0
    target[:, :, 1:8, 6] = 1.0
    # A prediction covers only the left target line. pred->target can look
    # good, but target->pred must expose the missing right line.
    logits = _line_logits(column=2).requires_grad_()
    terms = module.symmetric_distance_transform_loss(torch.sigmoid(logits), target, torch.tensor([[True, True]]))
    assert set(("loss", "target_to_pred", "pred_to_target")) <= set(terms)
    assert terms["target_to_pred"] > terms["pred_to_target"]
    one_way_mutation = terms["pred_to_target"]
    with pytest.raises(AssertionError):
        assert torch.allclose(terms["loss"], one_way_mutation, atol=1e-6)
    terms["loss"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    # A dense uncertain prediction is not zero distance to a thin target. This
    # catches an unnormalised log-sum-exp softmin that clamps negative values.
    uniform = module.symmetric_distance_transform_loss(
        torch.full_like(target, 0.5), target, torch.tensor([[True, True]])
    )
    assert uniform["target_to_pred"] > 0.01
    large_target = torch.zeros(1, 2, 45, 80)
    large_target[:, :, 8:39, 20] = 1.0
    large_target[:, :, 8:39, 59] = 1.0
    large_uniform = module.symmetric_distance_transform_loss(
        torch.full_like(large_target, 0.5), large_target, torch.tensor([[True, True]])
    )
    assert large_uniform["target_to_pred"] > 0.01


def test_p6_boundary_style_targets_and_loss_are_conditional() -> None:
    module = _module()
    assert hasattr(module, "build_boundary_style_targets"), "P6 needs lane attribute style targets"
    targets = module.build_boundary_style_targets(
        [
            {"side": "left", "attributes": {"lineStyle": "solid"}},
            {"side": "right", "attributes": {"lineStyle": "dashed"}},
            {"side": "center", "attributes": {"lineStyle": "unknown"}},
        ],
        device="cpu",
    )
    assert targets["targets"].tolist() == [[0, 1]]
    assert targets["valid_mask"].tolist() == [[True, True]]
    assert targets["valid_count"] == 2
    assigned = module.build_boundary_style_targets(
        [{"attributes": {"line_style": "solid"}}],
        assignments={0: "left"},
        device="cpu",
    )
    assert assigned["targets"].tolist() == [[0, -1]]
    assert assigned["valid_mask"].tolist() == [[True, False]]

    good = torch.tensor([[[7.0, -7.0, -7.0], [-7.0, 7.0, -7.0]]])
    bad = -good
    assert module.boundary_style_cross_entropy_loss(good, targets["targets"], targets["valid_mask"]).loss < module.boundary_style_cross_entropy_loss(
        bad, targets["targets"], targets["valid_mask"]
    ).loss
    bundle = module.road_grounding_loss_bundle(
        drivable_logits=torch.zeros(1, 3, 3, 3),
        drivable_targets=torch.zeros(1, 3, 3, 3),
        drivable_valid_mask=torch.tensor([[True, False, False]]),
        boundary_logits=torch.zeros(1, 2, 3, 3),
        boundary_targets=torch.zeros(1, 2, 3, 3),
        boundary_valid_mask=torch.tensor([[False, False]]),
        boundary_style_logits=good,
        boundary_style_targets=targets["targets"],
        boundary_style_valid_mask=targets["valid_mask"],
    )
    assert bundle["boundary_style"].active is True
    unknown = module.build_boundary_style_targets([{"side": "left", "attributes": {}}], device="cpu")
    inactive = module.boundary_style_cross_entropy_loss(
        torch.zeros(1, 2, 3), unknown["targets"], unknown["valid_mask"]
    )
    assert inactive.active is False and inactive.valid_count == 0
