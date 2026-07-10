from __future__ import annotations

import inspect
import math

import pytest
import torch

from fate_oia.models.mosaic_observable_predicates import MOSAICMultiPrototypeFactorBank


def _bank() -> MOSAICMultiPrototypeFactorBank:
    return MOSAICMultiPrototypeFactorBank(
        num_prototypes=(2, 3, 4, 2),
        region_priors=("upper_front", "front_center", "left_corridor", "right_corridor"),
        dim=8,
    )


def test_weighted_logsumexp_is_exact_and_not_a_prototype_mean() -> None:
    scores = torch.tensor([[[[[1.0]], [[3.0]], [[99.0]]]]])
    weights = torch.tensor([[[0.25, 0.75, 0.0]]])
    valid = torch.tensor([[True, True, False]])

    result = MOSAICMultiPrototypeFactorBank.aggregate_prototype_scores(scores, weights, valid)
    expected = math.log(0.25 * math.exp(1.0) + 0.75 * math.exp(3.0))
    assert result.shape == (1, 1, 1, 1)
    assert result.item() == pytest.approx(expected)
    assert result.item() != pytest.approx((1.0 + 3.0) / 2.0)


def test_prototype_bank_preserves_independent_scores_sparse_context_weights_and_diagnostics() -> None:
    torch.manual_seed(3)
    bank = _bank()
    context = torch.randn(2, 8, 12, 20)

    output = bank(context, prior_mode="content_only")

    assert output["prototype_scores"].shape == (2, 4, 4, 12, 20)
    assert output["prototype_weights"].shape == (2, 4, 4)
    assert output["coarse_scores"].shape == (2, 4, 12, 20)
    assert torch.allclose(output["prototype_weights"].sum(-1), torch.ones(2, 4), atol=1e-6)
    assert torch.count_nonzero(output["prototype_weights"][:, 0, 2:]) == 0
    assert torch.count_nonzero(output["prototype_weights"][:, 1, 3:]) == 0
    assert not torch.allclose(output["prototype_scores"][:, 0, 0], output["prototype_scores"][:, 0, 1])
    stats = output["prototype_stats"]
    assert set(stats) == {
        "prototype_occupancy",
        "prototype_effective_count",
        "prototype_pairwise_cosine",
        "dominant_prototype_rate",
        "dead_prototype_count",
    }
    assert stats["prototype_occupancy"].shape == (4, 4)
    assert stats["prototype_effective_count"].shape == (4,)


def test_prior_modes_are_isolated_and_prior_only_is_image_independent() -> None:
    torch.manual_seed(11)
    bank = _bank().eval()
    context_a = torch.randn(1, 8, 12, 20)
    context_b = torch.randn(1, 8, 12, 20) * 2.0 + 3.0

    content = bank(context_a, prior_mode="content_only")
    full = bank(context_a, prior_mode="full")
    prior_a = bank(context_a, prior_mode="prior_only")
    prior_b = bank(context_b, prior_mode="prior_only")

    assert not torch.allclose(content["coarse_scores"], full["coarse_scores"])
    assert not torch.allclose(content["coarse_scores"], prior_a["coarse_scores"])
    assert torch.allclose(prior_a["coarse_scores"], prior_b["coarse_scores"])
    assert torch.all(prior_a["prior_scale"] >= 0)
    assert torch.all(prior_a["prior_scale"] <= 0.20)


def test_every_valid_prototype_and_context_router_receive_distinct_finite_gradients() -> None:
    torch.manual_seed(17)
    bank = _bank()
    context = torch.randn(3, 8, 12, 20, requires_grad=True)
    output = bank(context, prior_mode="full")
    loss = output["coarse_scores"].square().mean()
    loss.backward()

    assert bank.prototypes.grad is not None
    valid_gradients = bank.prototypes.grad[bank.prototype_valid_mask]
    assert torch.isfinite(valid_gradients).all()
    assert torch.all(valid_gradients.abs().sum(-1) > 0)
    assert valid_gradients.std(dim=0).sum() > 0
    assert bank.context_router.weight.grad is not None
    assert bank.context_router.weight.grad.abs().sum() > 0
    assert context.grad is not None and context.grad.abs().sum() > 0


def test_prototype_bank_rejects_invalid_prior_mode_and_context_shape() -> None:
    bank = _bank()
    with pytest.raises(ValueError, match="prior_mode"):
        bank(torch.randn(1, 8, 12, 20), prior_mode="invalid")
    with pytest.raises(ValueError, match=r"\[B,D,12,20\]"):
        bank(torch.randn(1, 8, 10, 20), prior_mode="full")


def test_prototype_bank_source_never_averages_prototypes_before_matching() -> None:
    source = inspect.getsource(MOSAICMultiPrototypeFactorBank.forward)
    assert "prototypes.mean" not in source
    assert ".topk(" not in source
