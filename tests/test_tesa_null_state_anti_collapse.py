from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from fate_oia.datasets.meter_typed_targets import METERTypedTargetBuilder
from fate_oia.losses.meter_grounding_losses import (
    meter_grounding_loss,
    null_partition_calibration_loss,
)
from fate_oia.models.meter_signed_factors import TypedEvidenceStateHead


def _save_drivable(path: Path, *, center: bool = True) -> None:
    image = torch.zeros(12, 16, dtype=torch.uint8)
    if center:
        image[5:, 5:11] = 1
    else:
        image[5:, :3] = 1
    Image.fromarray(image.numpy(), mode="L").save(path)


def _builder() -> METERTypedTargetBuilder:
    return METERTypedTargetBuilder("configs/meter_factor_schema.yaml", grid_hw=(6, 8))


def test_factor_two_requires_center_corridor_and_complete_detection_source(tmp_path: Path) -> None:
    drivable = tmp_path / "drivable.png"
    _save_drivable(drivable)
    builder = _builder()

    unknown = builder.build({"drivable_map_path": str(drivable), "image_size": [16, 12], "objects": []})
    assert unknown["factor_anchor_valid"][2]
    assert not unknown["factor_source_complete"][2]
    assert not unknown["factor_state_valid"][2]

    clear = builder.build(
        {"drivable_map_path": str(drivable), "image_size": [16, 12], "source_complete": True, "objects": []}
    )
    assert clear["factor_source_complete"][2]
    assert clear["factor_state_valid"][2]
    assert clear["factor_state_target"][2].item() == 0
    assert clear["factor_present_valid"][2]
    assert not clear["factor_absent_valid"][2]

    occupied = builder.build(
        {
            "drivable_map_path": str(drivable),
            "image_size": [16, 12],
            "source_complete": True,
            "objects": [{"category": "car", "box2d": {"x1": 6, "y1": 6, "x2": 10, "y2": 11}}],
        }
    )
    assert occupied["factor_state_valid"][2]
    assert occupied["factor_state_target"][2].item() == 1
    assert occupied["factor_present_valid"][2]
    assert not occupied["factor_absent_valid"][2]

    side_only = tmp_path / "side_only.png"
    _save_drivable(side_only, center=False)
    invalid = builder.build(
        {"drivable_map_path": str(side_only), "image_size": [16, 12], "source_complete": True, "objects": []}
    )
    assert not invalid["factor_anchor_valid"][2]
    assert not invalid["factor_state_valid"][2]


def test_lateral_obstacle_factors_have_present_clear_or_unknown_semantics(tmp_path: Path) -> None:
    drivable = tmp_path / "drivable.png"
    _save_drivable(drivable)
    builder = _builder()

    unknown = builder.build({"drivable_map_path": str(drivable), "image_size": [16, 12], "objects": []})
    assert not unknown["factor_state_valid"][10]
    assert not unknown["factor_state_valid"][16]

    clear = builder.build(
        {"drivable_map_path": str(drivable), "image_size": [16, 12], "source_complete": True, "objects": []}
    )
    for factor in (10, 16):
        assert clear["factor_state_valid"][factor]
        assert clear["factor_state_target"][factor].item() == 1
        assert clear["factor_present_valid"][factor]
        assert not clear["factor_absent_valid"][factor]

    left_present = builder.build(
        {
            "drivable_map_path": str(drivable),
            "image_size": [16, 12],
            "objects": [{"category": "car", "box2d": {"x1": 1, "y1": 6, "x2": 5, "y2": 11}}],
        }
    )
    assert left_present["factor_state_valid"][10]
    assert left_present["factor_state_target"][10].item() == 0
    assert left_present["factor_present_valid"][10]
    assert not left_present["factor_absent_valid"][10]
    assert not left_present["factor_state_valid"][16]


def test_null_is_partition_calibrated_and_unknown_rows_have_zero_gradient() -> None:
    torch.manual_seed(0)
    head = TypedEvidenceStateHead(dim=8)
    nodes = torch.randn(1, 21, 8)
    patches = torch.randn(1, 3, 12, 8)
    output = head(nodes, patches, progress=1.0)
    assert output["factor_null_mass"].gt(0).all()
    assert output["factor_null_mass"].lt(1).all()
    assert output["factor_null_logit"].shape == (1, 21)

    null_mass = torch.tensor([[0.2, 0.8, 0.5]], requires_grad=True)
    present = torch.tensor([[True, False, False]])
    absent = torch.tensor([[False, True, False]])
    source = torch.ones(1, 3)
    loss = null_partition_calibration_loss(null_mass, present, absent, source)
    loss.backward()
    assert null_mass.grad[0, 0].abs() > 0
    assert null_mass.grad[0, 1].abs() > 0
    assert null_mass.grad[0, 2].eq(0)


def test_null_partition_is_safe_under_bf16_autocast() -> None:
    null_mass = torch.full((1, 3), 0.5, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss = null_partition_calibration_loss(
            null_mass,
            torch.tensor([[True, False, False]]),
            torch.tensor([[False, True, False]]),
            torch.ones(1, 3),
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(null_mass.grad).all()


def test_anchor_sparse_distribution_is_truly_sparse_after_exploration_decays() -> None:
    head = TypedEvidenceStateHead(dim=8, factor_dim=21)
    logits = torch.tensor([[12.0, -12.0, -12.0]], requires_grad=True)
    distribution = head._distribution(logits, progress=1.0)
    assert torch.allclose(distribution.sum(-1), torch.ones(1))
    assert distribution[0, 0] == 1.0
    assert distribution[0, 1] == 0.0
    assert distribution[0, 2] == 0.0


def test_grounding_weights_are_single_source_and_sum_exactly() -> None:
    torch.manual_seed(1)
    output = {
        "factor_anchor_map": torch.full((1, 21, 2), 0.5),
        "factor_state_logits": torch.zeros(1, 21, 3),
        "factor_observability_logit": torch.zeros(1, 21),
        "factor_observability": torch.full((1, 21), 0.5),
            "factor_null_mass": torch.full((1, 21), 0.5),
            "factor_typed_token": torch.zeros(1, 21, 2),
            "factor_state_prob": torch.full((1, 21, 3), 1 / 3),
            "factor_ontology_query": torch.zeros(21, 2),
            "factor_ontology_target": torch.zeros(21, 2),
            "state_ontology_query": torch.zeros(21, 3, 2),
            "state_ontology_target": torch.zeros(21, 3, 2),
            "factor_state_valid_mask": torch.ones(21, 3, dtype=torch.bool),
    }
    targets = {
        "factor_anchor_map": torch.full((1, 21, 2), 0.5),
        "factor_anchor_valid": torch.ones(1, 21, dtype=torch.bool),
        "factor_state_target": torch.zeros(1, 21, dtype=torch.long),
        "factor_state_valid": torch.ones(1, 21, dtype=torch.bool),
        "factor_present_valid": torch.ones(1, 21, dtype=torch.bool),
        "factor_absent_valid": torch.zeros(1, 21, dtype=torch.bool),
        "factor_observability": torch.ones(1, 21),
        "factor_observability_valid": torch.ones(1, 21, dtype=torch.bool),
        "factor_source_weight": torch.ones(1, 21),
    }
    weights = {"anchor": 0.2, "state": 0.3, "null": 0.4, "observability": 0.5, "discrimination": 0.6, "mirror": 0.7}
    result = meter_grounding_loss(
        output,
        targets,
        observability_tau=torch.full((21,), 0.5),
        weights=weights,
    )
    expected = sum(weights[name] * result[name] for name in weights)
    torch.testing.assert_close(result["total"], expected)
