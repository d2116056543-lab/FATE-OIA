from __future__ import annotations

from pathlib import Path

import torch

from fate_oia.models.mosaic_native_semantics import load_mosaic_schema_bundle
from fate_oia.models.mosaic_reason_decoder import MOSAICReasonDecoder


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _decoder(dim: int = 8) -> tuple[MOSAICReasonDecoder, dict]:
    bundle = load_mosaic_schema_bundle(CONFIG_ROOT)
    decoder = MOSAICReasonDecoder(
        [factor["name"] for factor in bundle["factors"]],
        bundle["states"],
        bundle["reason_observation"],
        dim=dim,
        highres_topk=16,
        midres_topk=8,
        self_attention_heads=2,
    )
    return decoder, bundle


def _pyramid(batch: int = 2, dim: int = 8) -> dict[str, torch.Tensor]:
    return {
        "F_hi": torch.randn(batch, dim, 45, 80),
        "F_mid": torch.randn(batch, dim, 23, 40),
        "F_ctx": torch.randn(batch, dim, 12, 20),
    }


def test_reason_decoder_builds_semantic_and_masked_visual_latent_reason_nodes() -> None:
    decoder, bundle = _decoder()
    batch = 2
    factor_features = torch.randn(batch, len(bundle["factors"]), 8)
    factor_masks = torch.rand(batch, len(bundle["factors"]), 45, 80)
    output = decoder(
        _pyramid(batch),
        factor_features,
        factor_masks,
        torch.rand(batch, len(bundle["states"])),
        torch.rand(batch, len(bundle["states"])),
    )
    assert set(output) == {
        "reason_logits_latent",
        "reason_nodes_semantic",
        "reason_nodes_visual",
        "reason_factor_masks",
        "reason_semantic_attention",
    }
    assert output["reason_logits_latent"].shape == (batch, 21)
    assert output["reason_nodes_semantic"].shape == (batch, 21, 8)
    assert output["reason_nodes_visual"].shape == (batch, 21, 8)
    assert output["reason_factor_masks"].shape == (batch, 21, 45, 80)
    assert torch.all((output["reason_factor_masks"] >= 0) & (output["reason_factor_masks"] <= 1))


def test_reason_mask_is_the_soft_union_of_mapped_factor_masks() -> None:
    decoder, bundle = _decoder()
    factor_count = len(bundle["factors"])
    masks = torch.zeros(1, factor_count, 45, 80)
    front_index = [factor["name"] for factor in bundle["factors"]].index("front_vehicle_visible")
    near_index = [factor["name"] for factor in bundle["factors"]].index("front_vehicle_near")
    masks[:, front_index, :10, :10] = 0.5
    masks[:, near_index, :10, :10] = 0.4
    output = decoder(
        _pyramid(batch=1),
        torch.randn(1, factor_count, 8),
        masks,
        torch.zeros(1, len(bundle["states"])),
        torch.zeros(1, len(bundle["states"])),
    )
    reason5 = output["reason_factor_masks"][0, 5, 0, 0]
    assert torch.allclose(reason5, torch.tensor(1.0 - (1.0 - 0.5) * (1.0 - 0.4)))


def test_factor_masks_change_visual_verification_and_all_reason_paths_receive_gradients() -> None:
    torch.manual_seed(43)
    decoder, bundle = _decoder()
    factor_count = len(bundle["factors"])
    state_count = len(bundle["states"])
    pyramid = _pyramid()
    factor_features = torch.randn(2, factor_count, 8, requires_grad=True)
    state_prob = torch.rand(2, state_count, requires_grad=True)
    state_uncertainty = torch.rand(2, state_count, requires_grad=True)
    zeros = torch.zeros(2, factor_count, 45, 80)
    ones = torch.ones_like(zeros)
    no_mask = decoder(pyramid, factor_features, zeros, state_prob, state_uncertainty)
    full_mask = decoder(pyramid, factor_features, ones, state_prob, state_uncertainty)
    assert not torch.allclose(no_mask["reason_nodes_visual"], full_mask["reason_nodes_visual"])
    full_mask["reason_logits_latent"].square().mean().backward()
    for parameter in (
        decoder.reason_queries,
        decoder.semantic_attention.in_proj_weight,
        decoder.visual_decoder.label_queries,
        decoder.classifier_weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_zero_state_contribution_cap_removes_state_information_exactly() -> None:
    torch.manual_seed(47)
    decoder, bundle = _decoder()
    pyramid = _pyramid(batch=1)
    factors = torch.randn(1, len(bundle["factors"]), 8)
    masks = torch.rand(1, len(bundle["factors"]), 45, 80)
    uncertainty = torch.zeros(1, len(bundle["states"]))
    zeros = decoder(
        pyramid, factors, masks, torch.zeros_like(uncertainty), uncertainty,
        state_contribution_cap=0.0,
    )["reason_logits_latent"]
    ones = decoder(
        pyramid, factors, masks, torch.ones_like(uncertainty), uncertainty,
        state_contribution_cap=0.0,
    )["reason_logits_latent"]
    assert torch.equal(zeros, ones)


def test_reason_state_contribution_is_bounded_by_point_two() -> None:
    decoder, bundle = _decoder()
    with torch.no_grad():
        output = decoder(
            _pyramid(batch=1),
            torch.randn(1, len(bundle["factors"]), 8),
            torch.rand(1, len(bundle["factors"]), 45, 80),
            torch.rand(1, len(bundle["states"])),
            torch.rand(1, len(bundle["states"])),
            state_contribution_cap=0.20,
        )
    assert torch.isfinite(output["reason_logits_latent"]).all()
    try:
        decoder(
            _pyramid(batch=1),
            torch.randn(1, len(bundle["factors"]), 8),
            torch.rand(1, len(bundle["factors"]), 45, 80),
            torch.rand(1, len(bundle["states"])),
            torch.rand(1, len(bundle["states"])),
            state_contribution_cap=0.21,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("reason state contribution exceeded the plan cap")
