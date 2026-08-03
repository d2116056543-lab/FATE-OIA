import math

import torch

from fate_oia.models.save_multiscale_field import (
    SAVEMultiscaleField,
    build_2d_sincos_position_embedding,
)


def test_fixed_2d_position_is_deterministic_and_non_trainable() -> None:
    position = build_2d_sincos_position_embedding(grid_hw=(45, 80), dim=384)
    repeated = build_2d_sincos_position_embedding(grid_hw=(45, 80), dim=384)

    assert position.shape == (1, 3600, 384)
    assert position.requires_grad is False
    assert torch.equal(position, repeated)
    assert not torch.equal(position[:, 0], position[:, 1])
    assert not torch.equal(position[:, 0], position[:, 80])

    field_builder = SAVEMultiscaleField()
    assert torch.equal(field_builder.detail_position, position)
    assert "detail_position" in dict(field_builder.named_buffers())
    assert "detail_position" not in dict(field_builder.named_parameters())


def test_fixed_2d_position_uses_row_major_sine_cosine_values() -> None:
    position = build_2d_sincos_position_embedding(grid_hw=(2, 3), dim=8)
    expected_origin = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    expected_next_column = torch.tensor(
        [
            0.0,
            0.0,
            1.0,
            1.0,
            math.sin(1.0),
            math.sin(0.01),
            math.cos(1.0),
            math.cos(0.01),
        ]
    )
    expected_next_row = torch.tensor(
        [
            math.sin(1.0),
            math.sin(0.01),
            math.cos(1.0),
            math.cos(0.01),
            0.0,
            0.0,
            1.0,
            1.0,
        ]
    )

    assert torch.allclose(position[0, 0], expected_origin)
    assert torch.allclose(position[0, 1], expected_next_column)
    assert torch.allclose(position[0, 3], expected_next_row)


def test_position_is_added_before_detail_norm_without_residual_contamination() -> None:
    field_builder = SAVEMultiscaleField(dim=8, input_dim=8, grid_hw=(2, 3))
    with torch.no_grad():
        field_builder.detail_projection_3.weight.zero_()
        field_builder.detail_projection_3.bias.zero_()
        field_builder.detail_projection_7.weight.zero_()
        field_builder.detail_projection_7.bias.zero_()
        field_builder.detail_output_projection.bias.fill_(2.0)
    patch_tokens = torch.randn(2, 3, 6, 8)

    output = field_builder(patch_tokens)
    expected = field_builder.detail_norm(field_builder.detail_position).expand(2, -1, -1)

    assert torch.allclose(output["detail_field"], expected)
    assert torch.equal(output["detail_residual"], torch.full_like(expected, 2.0))
