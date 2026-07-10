from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from fate_oia.models.mosaic_visual_pyramid import MOSAICVisualPyramid


def test_visual_pyramid_builds_all_three_required_scales() -> None:
    model = MOSAICVisualPyramid(input_dim=8, output_dim=6)
    tokens = torch.randn(2, 3, 45 * 80, 8)

    output = model(tokens)

    assert set(output) == {"F_hi", "F_mid", "F_ctx", "grid_hw"}
    assert output["F_hi"].shape == (2, 6, 45, 80)
    assert output["F_mid"].shape == (2, 6, 23, 40)
    assert output["F_ctx"].shape == (2, 6, 12, 20)
    assert output["grid_hw"] == (45, 80)


def test_visual_pyramid_uses_layers_3_7_11_in_order() -> None:
    model = MOSAICVisualPyramid(input_dim=4, output_dim=3)
    tokens = torch.stack(
        [
            torch.full((45 * 80, 4), 1.0),
            torch.full((45 * 80, 4), 2.0),
            torch.full((45 * 80, 4), 3.0),
        ],
        dim=0,
    ).unsqueeze(0)

    output = model(tokens)
    layer3 = tokens[:, 0].transpose(1, 2).reshape(1, 4, 45, 80)
    layer7 = tokens[:, 1].transpose(1, 2).reshape(1, 4, 45, 80)
    layer11 = tokens[:, 2].transpose(1, 2).reshape(1, 4, 45, 80)

    assert torch.allclose(output["F_hi"], model.proj_hi(layer3))
    assert torch.allclose(output["F_mid"], model.proj_mid(F.adaptive_avg_pool2d(layer7, (23, 40))))
    assert torch.allclose(output["F_ctx"], model.proj_ctx(F.adaptive_avg_pool2d(layer11, (12, 20))))


def test_local_depthwise_residual_is_zero_effect_but_trainable_at_initialization() -> None:
    model = MOSAICVisualPyramid(input_dim=5, output_dim=7)
    assert model.local_residual.groups == 7
    assert model.local_residual.kernel_size == (3, 3)
    assert torch.count_nonzero(model.local_residual.weight) == 0
    assert model.local_residual.bias is None

    tokens = torch.randn(1, 3, 45 * 80, 5, requires_grad=True)
    output = model(tokens)
    output["F_hi"].square().mean().backward()

    gradient = model.local_residual.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()


@pytest.mark.parametrize(
    "shape",
    [
        (2, 2, 45 * 80, 8),
        (2, 4, 45 * 80, 8),
        (2, 3, 3599, 8),
        (2, 3, 45 * 80, 7),
        (2, 3, 45, 80, 8),
    ],
)
def test_visual_pyramid_rejects_noncanonical_dense_field_shapes(shape: tuple[int, ...]) -> None:
    model = MOSAICVisualPyramid(input_dim=8, output_dim=6)
    with pytest.raises(ValueError, match=r"\[B,3,3600,input_dim\]"):
        model(torch.randn(*shape))


def test_visual_pyramid_rejects_non_float_inputs() -> None:
    model = MOSAICVisualPyramid(input_dim=8, output_dim=6)
    with pytest.raises(ValueError, match="floating-point"):
        model(torch.zeros(1, 3, 45 * 80, 8, dtype=torch.int64))
