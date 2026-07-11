from __future__ import annotations

import inspect

import pytest
import torch

from fate_oia.models.mosaic_observable_predicates import MOSAICObservablePredicateLayer


FACTORS = (
    {"name": "light", "type": "point", "num_prototypes": 2, "region_prior": "upper_front"},
    {"name": "car", "type": "object", "num_prototypes": 3, "region_prior": "front_center"},
    {"name": "lane", "type": "curve", "num_prototypes": 3, "region_prior": "left_corridor"},
    {"name": "drive", "type": "region", "num_prototypes": 4, "region_prior": "center_corridor"},
)


def _pyramid(batch: int = 2, dim: int = 8) -> dict[str, torch.Tensor]:
    return {
        "F_hi": torch.randn(batch, dim, 45, 80),
        "F_mid": torch.randn(batch, dim, 23, 40),
        "F_ctx": torch.randn(batch, dim, 12, 20),
    }


def test_observable_layer_returns_complete_semantically_exact_outputs() -> None:
    torch.manual_seed(19)
    layer = MOSAICObservablePredicateLayer(FACTORS, dim=8)
    output = layer(_pyramid(), prior_mode="full")

    expected_keys = {
        "factor_features",
        "factor_presence_logits",
        "factor_presence_prob",
        "factor_visibility_logits",
        "factor_visibility_prob",
        "factor_positive_evidence",
        "factor_negative_evidence",
        "factor_uncertainty",
        "factor_soft_masks",
        "prototype_weights",
        "anchor_coordinates",
        "sampling_coordinates",
        "prior_scale",
        "measurement_stats",
    }
    assert set(output) == expected_keys
    assert output["factor_features"].shape == (2, 4, 8)
    assert output["factor_presence_logits"].shape == (2, 4)
    assert output["factor_visibility_logits"].shape == (2, 4)
    assert output["factor_soft_masks"].shape == (2, 4, 45, 80)
    assert output["prototype_weights"].shape == (2, 4, 4)
    assert output["anchor_coordinates"].shape == (2, 4, 2, 2)
    assert output["sampling_coordinates"].shape == (2, 4, 2, 4, 12, 2)
    visibility = output["factor_visibility_prob"]
    presence = output["factor_presence_prob"]
    assert torch.allclose(output["factor_positive_evidence"], visibility * presence)
    assert torch.allclose(output["factor_negative_evidence"], visibility * (1.0 - presence))
    assert torch.all(output["factor_uncertainty"] >= 0)
    assert torch.all(output["factor_uncertainty"] <= 1)


def test_low_visibility_suppresses_both_positive_and_negative_evidence() -> None:
    layer = MOSAICObservablePredicateLayer(FACTORS, dim=8)
    with torch.no_grad():
        layer.visibility_head.weight.zero_()
        layer.visibility_head.bias.fill_(-20.0)
        layer.presence_head.weight.zero_()
        layer.presence_head.bias.copy_(torch.tensor([20.0, -20.0, 20.0, -20.0]))
    output = layer(_pyramid(batch=1), prior_mode="content_only")
    assert output["factor_positive_evidence"].max() < 1e-7
    assert output["factor_negative_evidence"].max() < 1e-7


def test_soft_anchors_and_all_measurement_paths_receive_gradients() -> None:
    torch.manual_seed(23)
    layer = MOSAICObservablePredicateLayer(FACTORS, dim=8, prior_dropout=0.0)
    pyramid = _pyramid()
    for value in pyramid.values():
        value.requires_grad_()
    output = layer(pyramid, prior_mode="full")
    loss = (
        output["factor_presence_logits"].square().mean()
        + output["factor_visibility_logits"].square().mean()
        + output["anchor_coordinates"].square().mean()
    )
    loss.backward()

    parameters = (
        layer.prototype_bank.prototypes,
        layer.prototype_bank.context_router.weight,
        layer.prototype_bank.prior_scale_raw,
        layer.anchor_temperature_raw,
        layer.typed_attention.point_offset_delta,
        layer.typed_attention.curve_tangent_raw,
        layer.typed_attention.region_extent_raw,
        layer.presence_head.weight,
        layer.visibility_head.weight,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_observable_layer_has_two_distinct_differentiable_anchors_without_hard_topk() -> None:
    torch.manual_seed(29)
    layer = MOSAICObservablePredicateLayer(FACTORS, dim=8)
    output = layer(_pyramid(batch=3), prior_mode="content_only")
    anchors = output["anchor_coordinates"]
    assert not torch.allclose(anchors[:, :, 0], anchors[:, :, 1])
    source = inspect.getsource(MOSAICObservablePredicateLayer)
    assert ".topk(" not in source


@pytest.mark.parametrize("prior_mode", ["full", "content_only", "prior_only"])
def test_observable_layer_supports_all_audit_prior_modes(prior_mode: str) -> None:
    layer = MOSAICObservablePredicateLayer(FACTORS, dim=8).eval()
    output = layer(_pyramid(batch=3), prior_mode=prior_mode)
    assert output["factor_features"].shape == (3, 4, 8)
    assert output["factor_presence_prob"].shape == (3, 4)
    assert output["factor_visibility_prob"].shape == (3, 4)


def test_prior_only_is_independent_of_high_mid_and_context_image_content() -> None:
    layer = MOSAICObservablePredicateLayer(FACTORS, dim=8).eval()
    pyramid_a = _pyramid(batch=1)
    pyramid_b = {key: torch.randn_like(value) * 3.0 + 2.0 for key, value in pyramid_a.items()}
    output_a = layer(pyramid_a, prior_mode="prior_only")
    output_b = layer(pyramid_b, prior_mode="prior_only")
    for key in (
        "factor_features",
        "factor_presence_logits",
        "factor_visibility_logits",
        "anchor_coordinates",
        "factor_soft_masks",
    ):
        assert torch.allclose(output_a[key], output_b[key])


def test_mid_level_features_are_active_and_receive_gradients() -> None:
    torch.manual_seed(47)
    layer = MOSAICObservablePredicateLayer(FACTORS, dim=8, prior_dropout=0.0)
    pyramid = _pyramid(batch=1)
    changed = {key: value.clone() for key, value in pyramid.items()}
    changed["F_mid"] = changed["F_mid"] + torch.randn_like(changed["F_mid"]) * 2.0
    baseline = layer(pyramid, prior_mode="content_only")["factor_features"]
    modified = layer(changed, prior_mode="content_only")["factor_features"]
    assert not torch.allclose(baseline, modified)

    pyramid["F_mid"].requires_grad_()
    layer(pyramid, prior_mode="content_only")["factor_presence_logits"].sum().backward()
    assert pyramid["F_mid"].grad is not None
    assert pyramid["F_mid"].grad.abs().sum() > 0


def test_observable_layer_accepts_direct_bfloat16_pyramid_inputs() -> None:
    layer = MOSAICObservablePredicateLayer(FACTORS, dim=8)
    pyramid = {key: value.to(torch.bfloat16) for key, value in _pyramid(batch=1).items()}
    output = layer(pyramid, prior_mode="content_only")
    assert torch.isfinite(output["factor_presence_logits"]).all()
    assert torch.isfinite(output["factor_visibility_logits"]).all()


def test_observable_layer_rejects_incomplete_pyramid() -> None:
    layer = MOSAICObservablePredicateLayer(FACTORS, dim=8)
    with pytest.raises(ValueError, match="F_hi/F_mid/F_ctx"):
        layer({"F_hi": torch.randn(1, 8, 45, 80)}, prior_mode="full")
