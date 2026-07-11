from __future__ import annotations

from pathlib import Path

import torch

from fate_oia.losses.mosaic_reason_observation_losses import fixed_propensity_observation_loss


def test_fixed_propensity_observation_loss_penalizes_all_on_without_hard_latent_targets() -> None:
    observed = torch.tensor([[1.0, 0.0]])
    good = torch.tensor([[0.8, 0.1]], requires_grad=True)
    all_on = torch.tensor([[0.8, 0.8]], requires_grad=True)
    good_loss = fixed_propensity_observation_loss(good, observed)
    all_on_loss = fixed_propensity_observation_loss(all_on, observed)
    assert good_loss < all_on_loss
    good_loss.backward()
    assert good.grad is not None and torch.isfinite(good.grad).all()

from fate_oia.models.mosaic_native_semantics import load_mosaic_schema_bundle
from fate_oia.models.mosaic_selective_observation import MOSAICSelectiveObservationModel


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"


def _model() -> tuple[MOSAICSelectiveObservationModel, dict]:
    bundle = load_mosaic_schema_bundle(CONFIG_ROOT)
    model = MOSAICSelectiveObservationModel(
        [factor["name"] for factor in bundle["factors"]],
        bundle["reason_observation"],
    )
    return model, bundle


def test_propensity_is_group_shared_bounded_and_uses_only_stopgrad_observability() -> None:
    model, bundle = _model()
    batch = 3
    visibility = torch.rand(batch, len(bundle["factors"]), requires_grad=True)
    uncertainty = torch.rand(batch, len(bundle["factors"]), requires_grad=True)
    logits = torch.randn(batch, 21, requires_grad=True)
    labels = torch.randint(0, 2, (batch, 21)).float()

    output = model(logits, labels, visibility, uncertainty)

    propensity = output["reason_propensity"]
    assert propensity.shape == (batch, 21)
    assert propensity.min() >= 0.20
    assert propensity.max() <= 0.95
    propensity.sum().backward(retain_graph=True)
    assert visibility.grad is None
    assert uncertainty.grad is None
    assert logits.grad is None
    assert model.group_bias.grad is not None
    assert model.group_bias.numel() == 4


def test_observation_probability_and_exact_zero_posterior_match_closed_form() -> None:
    model, bundle = _model()
    with torch.no_grad():
        model.group_bias.zero_()
        model.raw_visibility_weight.fill_(-20.0)
        model.raw_uncertainty_weight.fill_(-20.0)
        model.false_positive_raw.fill_(-2.0)
    logits = torch.tensor([[0.7] * 21])
    labels = torch.zeros(1, 21)
    output = model(
        logits,
        labels,
        torch.full((1, len(bundle["factors"])), 0.5),
        torch.full((1, len(bundle["factors"])), 0.5),
    )
    p_star = torch.sigmoid(logits)
    pi = output["reason_propensity"]
    epsilon = output["reason_false_positive_rate"]
    expected_observed = pi * p_star + epsilon * (1.0 - p_star)
    expected_q_zero = p_star * (1.0 - pi) / (
        p_star * (1.0 - pi) + (1.0 - p_star) * (1.0 - epsilon)
    )
    assert torch.allclose(output["reason_observation_prob"], expected_observed, atol=1e-6)
    assert torch.allclose(output["reason_latent_posterior"], expected_q_zero, atol=1e-6)


def test_observed_positive_posterior_is_exactly_one_and_posterior_is_detached() -> None:
    model, bundle = _model()
    labels = torch.zeros(2, 21)
    labels[:, [0, 5, 14]] = 1
    output = model(
        torch.randn(2, 21, requires_grad=True),
        labels,
        torch.rand(2, len(bundle["factors"])),
        torch.rand(2, len(bundle["factors"])),
    )
    assert torch.equal(output["reason_latent_posterior"][labels.bool()], torch.ones_like(labels[labels.bool()]))
    assert output["reason_latent_posterior"].requires_grad is False


def test_reason_labels_never_change_propensity_features() -> None:
    model, bundle = _model()
    logits = torch.randn(2, 21)
    visibility = torch.rand(2, len(bundle["factors"]))
    uncertainty = torch.rand(2, len(bundle["factors"]))
    zeros = model(logits, torch.zeros(2, 21), visibility, uncertainty)
    ones = model(logits, torch.ones(2, 21), visibility, uncertainty)
    assert torch.equal(zeros["reason_propensity"], ones["reason_propensity"])


def test_synthetic_missing_positive_mask_hides_only_observed_positives() -> None:
    model, _ = _model()
    labels = torch.zeros(10, 21)
    labels[:, :10] = 1
    generator = torch.Generator().manual_seed(20260710)
    hidden_labels, hidden_mask = model.hide_observed_positives(labels, hide_fraction=0.20, generator=generator)
    assert torch.all(hidden_mask <= labels.bool())
    assert torch.equal(hidden_labels[hidden_mask], torch.zeros_like(hidden_labels[hidden_mask]))
    assert torch.equal(hidden_labels[~hidden_mask], labels[~hidden_mask])
    assert 10 <= int(hidden_mask.sum()) <= 30


def test_false_positive_rates_are_label_specific_and_never_exceed_config_bounds() -> None:
    model, bundle = _model()
    output = model(
        torch.zeros(1, 21),
        torch.zeros(1, 21),
        torch.zeros(1, len(bundle["factors"])),
        torch.zeros(1, len(bundle["factors"])),
    )
    epsilon = output["reason_false_positive_rate"]
    assert epsilon.shape == (21,)
    expected_max = torch.tensor(
        [bundle["reason_observation"][reason_id]["false_positive_max"] for reason_id in range(21)]
    )
    assert torch.all(epsilon >= 0)
    assert torch.all(epsilon <= expected_max)
