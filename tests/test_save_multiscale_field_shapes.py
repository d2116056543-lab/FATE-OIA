import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from fate_oia.models.acpr_dino_field import ACPRDinoFieldExtractor
from fate_oia.models.meter_calalign_foundation import METERCalAlignFoundation
from fate_oia.models.save_multiscale_field import SAVEMultiscaleField


def test_save_multiscale_fields_preserve_full_dino_patch_contract() -> None:
    torch.manual_seed(0)
    foundation = METERCalAlignFoundation(use_mock_dino=True)
    images = torch.randn(1, 3, 360, 640)

    encoded = foundation.encode_images(images)
    patch_tokens = encoded["patch_tokens_by_layer"]

    assert foundation.ordinary_dino_calls == 1
    assert tuple(foundation.selected_layers) == (3, 7, 11)
    assert patch_tokens.shape == (1, 3, 3600, 384)
    assert patch_tokens.requires_grad is False
    assert foundation.dino.training is False
    assert all(not parameter.requires_grad for parameter in foundation.dino.parameters())

    field_builder = SAVEMultiscaleField()
    output = field_builder(patch_tokens)

    assert set(output) == {
        "global_field",
        "detail_field",
        "detail_position",
        "detail_residual",
    }
    assert output["global_field"].shape == (1, 3600, 384)
    assert output["detail_field"].shape == (1, 3600, 384)
    assert output["detail_position"].shape == (1, 3600, 384)
    assert output["detail_residual"].shape == (1, 3600, 384)
    assert output["detail_position"].requires_grad is False

    loss = output["global_field"].square().mean() + output["detail_field"].square().mean()
    loss.backward()
    assert field_builder.global_projection.weight.grad is not None
    assert field_builder.detail_projection_3.weight.grad is not None
    assert field_builder.detail_projection_7.weight.grad is not None
    assert all(parameter.grad is None for parameter in foundation.dino.parameters())


def test_canonical_fields_follow_exact_layer_equations() -> None:
    torch.manual_seed(1)
    field_builder = SAVEMultiscaleField(dim=8, input_dim=8, grid_hw=(2, 3))
    patch_tokens = torch.randn(2, 3, 6, 8)
    with torch.no_grad():
        field_builder.detail_output_projection.weight.fill_(0.125)
        field_builder.detail_output_projection.bias.fill_(0.25)

    output = field_builder(patch_tokens)
    expected_global = field_builder.global_norm(
        F.linear(
            patch_tokens[:, 2],
            field_builder.global_projection.weight,
            field_builder.global_projection.bias,
        )
    )
    expected_detail = field_builder.detail_norm(
        F.linear(
            patch_tokens[:, 0],
            field_builder.detail_projection_3.weight,
            field_builder.detail_projection_3.bias,
        )
        + F.linear(
            patch_tokens[:, 1],
            field_builder.detail_projection_7.weight,
            field_builder.detail_projection_7.bias,
        )
        + field_builder.detail_position
    )
    expected_residual = F.linear(
        expected_detail,
        field_builder.detail_output_projection.weight,
        field_builder.detail_output_projection.bias,
    )

    assert torch.allclose(output["global_field"], expected_global)
    assert torch.allclose(output["detail_field"], expected_detail)
    assert torch.allclose(output["detail_residual"], expected_residual)
    assert torch.count_nonzero(output["detail_residual"]) > 0


def test_detail_residual_is_zero_initialized_with_live_auxiliary_gradient() -> None:
    torch.manual_seed(2)
    field_builder = SAVEMultiscaleField(dim=8, input_dim=8, grid_hw=(2, 3))
    patch_tokens = torch.randn(2, 3, 6, 8)

    assert torch.count_nonzero(field_builder.detail_output_projection.weight) == 0
    assert torch.count_nonzero(field_builder.detail_output_projection.bias) == 0
    output = field_builder(patch_tokens)
    assert torch.count_nonzero(output["detail_residual"]) == 0

    auxiliary_target = torch.ones_like(output["detail_residual"])
    F.mse_loss(output["detail_residual"], auxiliary_target).backward()
    weight_grad = field_builder.detail_output_projection.weight.grad
    bias_grad = field_builder.detail_output_projection.bias.grad
    assert weight_grad is not None and torch.count_nonzero(weight_grad) > 0
    assert bias_grad is not None and torch.count_nonzero(bias_grad) > 0


def test_layer_projections_are_independent_and_layer_order_is_fixed() -> None:
    field_builder = SAVEMultiscaleField(dim=8, input_dim=8, grid_hw=(2, 3))
    projections = (
        field_builder.projection_3,
        field_builder.projection_7,
        field_builder.projection_11,
    )

    assert len({projection.weight.data_ptr() for projection in projections}) == 3
    assert len({projection.bias.data_ptr() for projection in projections}) == 3
    with pytest.raises(ValueError, match=r"layers \(3, 7, 11\)"):
        SAVEMultiscaleField(selected_layers=(2, 7, 11))


def test_full_resolution_calls_are_uncached_and_token_local() -> None:
    field_builder = SAVEMultiscaleField(dim=8, input_dim=8)
    with torch.no_grad():
        for projection in (
            field_builder.projection_3,
            field_builder.projection_7,
            field_builder.projection_11,
        ):
            projection.weight.copy_(torch.eye(8))
            projection.bias.zero_()

    baseline_tokens = torch.randn(1, 3, 3600, 8)
    baseline = field_builder(baseline_tokens)
    baseline_global = baseline["global_field"].clone()
    baseline_detail = baseline["detail_field"].clone()

    layer_3_and_11 = baseline_tokens.clone()
    layer_3_and_11[0, 0, 137, 0] += 1.0
    layer_3_and_11[0, 2, 2718, 1] += 1.0
    changed = field_builder(layer_3_and_11)

    global_changed_tokens = (
        changed["global_field"] - baseline_global
    ).abs().amax(dim=-1).nonzero(as_tuple=False)
    detail_changed_tokens = (
        changed["detail_field"] - baseline_detail
    ).abs().amax(dim=-1).nonzero(as_tuple=False)
    assert global_changed_tokens.tolist() == [[0, 2718]]
    assert detail_changed_tokens.tolist() == [[0, 137]]

    layer_7_only = baseline_tokens.clone()
    layer_7_only[0, 1, 512, 2] += 1.0
    changed_again = field_builder(layer_7_only)
    assert torch.equal(baseline["global_field"], baseline_global)
    assert torch.equal(baseline["detail_field"], baseline_detail)
    assert torch.equal(changed_again["global_field"], baseline_global)
    detail_changed_again = (
        changed_again["detail_field"] - baseline_detail
    ).abs().amax(dim=-1).nonzero(as_tuple=False)
    assert detail_changed_again.tolist() == [[0, 512]]
    assert changed_again["detail_field"].shape[1] == 3600


@pytest.mark.parametrize(
    ("shape", "message"),
    (
        ((1, 2, 6, 8), "exactly 3 selected layers"),
        ((1, 3, 5, 8), "exactly 6 patch tokens"),
        ((1, 3, 6, 7), "channel width 8"),
    ),
)
def test_rejects_malformed_multiscale_input(
    shape: tuple[int, ...], message: str
) -> None:
    field_builder = SAVEMultiscaleField(dim=8, input_dim=8, grid_hw=(2, 3))

    with pytest.raises(ValueError, match=message):
        field_builder(torch.randn(shape))


def test_real_dino_selected_layer_probe() -> None:
    weights_value = os.environ.get("SAVE_REAL_DINO_WEIGHTS")
    if not weights_value:
        pytest.skip("set SAVE_REAL_DINO_WEIGHTS to run the real-DINO probe")
    weights_path = Path(weights_value)
    assert weights_path.is_file(), f"real-DINO weights not found: {weights_path}"

    extractor = ACPRDinoFieldExtractor(
        selected_layers=(3, 7, 11),
        pretrained_weights=str(weights_path),
        use_mock_dino=False,
        freeze_backbone=True,
    )
    captured_layers: dict[int, torch.Tensor] = {}
    block_calls = [0 for _ in extractor.backbone.blocks]

    def make_block_hook(index: int):
        def record_block(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            value: torch.Tensor,
        ) -> None:
            block_calls[index] += 1
            layer = index + 1
            if layer in extractor.selected_layers:
                captured_layers[layer] = value.detach()

        return record_block

    block_handles = [
        block.register_forward_hook(make_block_hook(index))
        for index, block in enumerate(extractor.backbone.blocks)
    ]
    images = torch.randn(1, 3, 360, 640, requires_grad=True)
    with patch.object(
        extractor.backbone,
        "prepare_tokens",
        wraps=extractor.backbone.prepare_tokens,
    ) as prepare_tokens:
        try:
            output = extractor(images)
        finally:
            for handle in block_handles:
                handle.remove()

    assert prepare_tokens.call_count == 1
    assert block_calls == [1] * len(extractor.backbone.blocks)
    assert extractor.selected_layers == (3, 7, 11)
    assert extractor.backbone.training is False
    assert all(not parameter.requires_grad for parameter in extractor.parameters())
    patch_layers = output["patch_tokens_by_layer"]
    assert patch_layers.shape == (1, 3, 3600, 384)
    assert patch_layers.requires_grad is False
    assert images.grad is None
    assert tuple(captured_layers) == (3, 7, 11)
    with torch.no_grad():
        for output_index, layer in enumerate(extractor.selected_layers):
            expected = extractor.backbone.norm(captured_layers[layer])[:, 1:]
            assert torch.allclose(patch_layers[:, output_index], expected)
    assert torch.equal(output["patch_tokens_last"], patch_layers[:, -1])
