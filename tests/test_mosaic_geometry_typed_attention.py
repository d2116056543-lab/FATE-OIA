from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from fate_oia.models.mosaic_geometry_typed_attention import MOSAICGeometryTypedAttention


FACTOR_TYPES = ("point", "object", "curve", "region")


def test_typed_attention_vectorizes_one_grid_sample_call_per_geometry_group(monkeypatch) -> None:
    calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    original_grid_sample = F.grid_sample

    def recording_grid_sample(input_tensor, grid, *args, **kwargs):
        calls.append((tuple(input_tensor.shape), tuple(grid.shape)))
        return original_grid_sample(input_tensor, grid, *args, **kwargs)

    monkeypatch.setattr(F, "grid_sample", recording_grid_sample)
    module = MOSAICGeometryTypedAttention(FACTOR_TYPES, dim=8, anchors_per_factor=2, heads=4)
    feature_map = torch.randn(2, 8, 45, 80)
    anchors = torch.zeros(2, 4, 2, 2)

    output = module(feature_map, anchors)

    assert len(calls) == 3
    assert all(input_shape == (2, 8, 45, 80) for input_shape, _ in calls)
    assert sorted(grid_shape[1] for _, grid_shape in calls) == [8, 8, 16]
    # V4's fine transport uses IC-DOR's 4/16/12 budget by default.
    assert output["sampled_features"].shape == (2, 4, 2, 4, 16, 8)
    assert output["sampling_coordinates"].shape == (2, 4, 2, 4, 16, 2)
    assert output["sample_valid_mask"].shape == (4, 16)
    assert output["sample_valid_mask"][0].sum() == 4
    assert output["sample_valid_mask"][1].sum() == 4
    assert output["sample_valid_mask"][2].sum() == 16
    assert output["sample_valid_mask"][3].sum() == 12
    assert torch.count_nonzero(output["sampled_features"][:, :2, :, :, 4:]) == 0


def test_typed_attention_coordinates_and_all_geometry_parameters_receive_gradients() -> None:
    torch.manual_seed(7)
    module = MOSAICGeometryTypedAttention(FACTOR_TYPES, dim=6, anchors_per_factor=2, heads=4)
    feature_map = torch.randn(2, 6, 45, 80, requires_grad=True)
    anchors = torch.zeros(2, 4, 2, 2, requires_grad=True)

    output = module(feature_map, anchors)
    coordinates = output["sampling_coordinates"]
    assert coordinates.min() >= -1.0
    assert coordinates.max() <= 1.0
    loss = output["sampled_features"].square().mean() + 0.01 * coordinates.square().mean()
    loss.backward()

    required_parameters = (
        module.point_offset_delta,
        module.curve_tangent_raw,
        module.curve_longitudinal_delta,
        module.curve_lateral_delta,
        module.region_extent_raw,
        module.region_grid_delta,
    )
    for parameter in required_parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert anchors.grad is not None and anchors.grad.abs().sum() > 0
    assert feature_map.grad is not None and feature_map.grad.abs().sum() > 0


def test_point_curve_and_region_sampling_are_geometrically_distinct() -> None:
    module = MOSAICGeometryTypedAttention(("point", "curve", "region"), dim=4)
    output = module(torch.randn(1, 4, 45, 80), torch.zeros(1, 3, 2, 2))
    coordinates = output["sampling_coordinates"][0]

    point = coordinates[0, :, :, :4]
    curve = coordinates[1]
    region = coordinates[2]
    assert point.std(dim=-2).mean() > 0
    assert curve[..., 1].std() > curve[..., 0].std()
    assert region[..., 0].std() > 0 and region[..., 1].std() > 0
    assert not torch.allclose(curve, region)


def test_typed_attention_source_has_no_per_factor_grid_sample_loop() -> None:
    source = inspect.getsource(MOSAICGeometryTypedAttention.forward)
    assert "for factor" not in source
    assert ".grid_sample" not in source


@pytest.mark.parametrize(
    ("factor_types", "error"),
    [
        (("point", "unknown"), "unsupported factor type"),
        ((), "at least one factor"),
    ],
)
def test_typed_attention_rejects_invalid_factor_contract(factor_types, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        MOSAICGeometryTypedAttention(factor_types, dim=8)


@pytest.mark.parametrize(
    ("feature_shape", "anchor_shape"),
    [
        ((2, 8, 44, 80), (2, 4, 2, 2)),
        ((2, 8, 45, 80), (2, 3, 2, 2)),
        ((2, 8, 45, 80), (2, 4, 1, 2)),
        ((2, 7, 45, 80), (2, 4, 2, 2)),
    ],
)
def test_typed_attention_rejects_noncanonical_shapes(feature_shape, anchor_shape) -> None:
    module = MOSAICGeometryTypedAttention(FACTOR_TYPES, dim=8, anchors_per_factor=2)
    with pytest.raises(ValueError, match="shape contract"):
        module(torch.randn(*feature_shape), torch.zeros(*anchor_shape))


def test_typed_attention_runs_bfloat16_features_with_fp32_sampling_and_backward() -> None:
    module = MOSAICGeometryTypedAttention(FACTOR_TYPES, dim=4)
    feature_map = torch.randn(1, 4, 45, 80, dtype=torch.bfloat16, requires_grad=True)
    anchors = torch.zeros(1, 4, 2, 2, dtype=torch.float32, requires_grad=True)

    output = module(feature_map, anchors)
    assert output["sampled_features"].dtype == torch.bfloat16
    assert output["sampling_coordinates"].dtype == torch.float32
    output["sampled_features"].float().square().mean().backward()
    assert feature_map.grad is not None and torch.isfinite(feature_map.grad).all()
    assert anchors.grad is not None and torch.isfinite(anchors.grad).all()


def test_bfloat16_path_reuses_one_fp32_feature_map_for_all_geometry_groups(monkeypatch) -> None:
    input_storage_pointers: list[int] = []
    original_grid_sample = F.grid_sample

    def recording_grid_sample(input_tensor, grid, *args, **kwargs):
        input_storage_pointers.append(input_tensor.untyped_storage().data_ptr())
        return original_grid_sample(input_tensor, grid, *args, **kwargs)

    monkeypatch.setattr(F, "grid_sample", recording_grid_sample)
    module = MOSAICGeometryTypedAttention(FACTOR_TYPES, dim=4)
    feature_map = torch.randn(1, 4, 45, 80, dtype=torch.bfloat16, requires_grad=True)
    output = module(feature_map, torch.zeros(1, 4, 2, 2))
    assert len(input_storage_pointers) == 3
    assert len(set(input_storage_pointers)) == 1
    output["sampled_features"].float().sum().backward()


@pytest.mark.parametrize(
    "overrides",
    [
        {"anchors_per_factor": 1},
        {"heads": 1},
        {"point_samples": 7},
        {"curve_samples": 11},
        {"region_samples": 11},
    ],
)
def test_typed_attention_enforces_the_formal_fixed_sampling_budget(overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="fixed sampling contract"):
        MOSAICGeometryTypedAttention(FACTOR_TYPES, dim=8, **overrides)
