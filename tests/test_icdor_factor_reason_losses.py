from __future__ import annotations

import torch
from torch.nn import functional as F

from fate_oia.losses.mosaic_icdor_factor_losses import (
    factor_presence_visibility_losses,
    factor_prototype_regularization,
    factor_view_consistency_loss,
)
from fate_oia.losses.mosaic_icdor_reason_losses import build_synthetic_hidden_positive_mask, selective_observation_losses
from fate_oia.losses.mosaic_icdor_transport_losses import MOSAICDetachedPosteriorQueue


def test_icdor_factor_unknowns_are_masked_not_converted_to_negatives() -> None:
    presence = torch.randn(3, 5, requires_grad=True)
    visibility = torch.randn(3, 5, requires_grad=True)
    target = torch.zeros(3, 5)
    known = torch.zeros(3, 5, dtype=torch.bool)
    losses = factor_presence_visibility_losses(presence, visibility, target, target, known, known)
    assert losses["loss_factor_presence"].item() == 0.0
    assert losses["loss_factor_visibility"].item() == 0.0
    losses["loss_factor_total"].backward()
    assert torch.count_nonzero(presence.grad) == 0
    assert torch.count_nonzero(visibility.grad) == 0


def test_icdor_reason_posterior_losses_and_ring_queue_are_detached() -> None:
    latent = torch.randn(4, 21, requires_grad=True)
    observed = torch.randint(0, 2, (4, 21), dtype=torch.float32)
    probability = torch.sigmoid(torch.randn(4, 21))
    posterior = torch.rand(4, 21)
    losses = selective_observation_losses(
        latent,
        observed,
        probability,
        posterior,
        reason_propensity=torch.full((4, 21), 0.5),
        factor_route_support=torch.rand(4, 21),
        escape_weight=torch.rand(4, 21),
    )
    assert torch.isfinite(losses["loss_reason_selective_total"])
    losses["loss_reason_selective_total"].backward()
    assert latent.grad is not None

    queue = MOSAICDetachedPosteriorQueue(capacity=8, label_count=21, device="cpu")
    queue.enqueue(torch.randn(4, 21, requires_grad=True), posterior, torch.arange(4))
    assert queue.size == 4
    assert queue.logits.requires_grad is False
    queue.enqueue(torch.randn(6, 21), torch.rand(6, 21), torch.arange(6, 12))
    assert queue.size == 8
    assert queue.write_index == 2


def test_icdor_selective_observation_uses_logits_for_autocast_safe_nll() -> None:
    """The trainer must preserve the observation logits emitted by the model."""
    latent = torch.randn(2, 21, requires_grad=True)
    observation_logits = torch.randn(2, 21, requires_grad=True)
    observed = torch.randint(0, 2, (2, 21), dtype=torch.float32)
    losses = selective_observation_losses(
        latent,
        observed,
        torch.sigmoid(observation_logits),
        torch.full((2, 21), 0.5),
        reason_observation_logits=observation_logits,
        reason_propensity=torch.full((2, 21), 0.5),
        factor_route_support=torch.full((2, 21), 0.5),
        escape_weight=torch.zeros((2, 21)),
    )
    assert torch.allclose(
        losses["loss_reason_observation_nll"],
        F.binary_cross_entropy_with_logits(observation_logits, observed),
    )
    losses["loss_reason_selective_total"].backward()
    assert observation_logits.grad is not None
    assert torch.isfinite(observation_logits.grad).all()


def test_icdor_synthetic_hidden_recovery_really_hides_observed_positives() -> None:
    observed = torch.zeros(3, 21)
    observed[:, :4] = 1.0
    hidden = build_synthetic_hidden_positive_mask(
        observed,
        hide_fraction=0.5,
        generator=torch.Generator().manual_seed(9),
    )
    assert hidden.any()
    assert torch.all(~hidden | observed.bool())
    assert torch.all((observed.bool() & ~hidden).sum(dim=1) >= 1)
    latent = torch.zeros_like(observed, requires_grad=True)
    losses = selective_observation_losses(
        latent,
        observed.masked_fill(hidden, 0.0),
        torch.full_like(observed, 0.5),
        torch.full_like(observed, 0.5),
        reason_propensity=torch.full_like(observed, 0.5),
        factor_route_support=torch.full_like(observed, 0.5),
        escape_weight=torch.zeros_like(observed),
        synthetic_hidden_positive_mask=hidden,
        observed_valid_mask=~hidden,
    )
    # Hidden truths are evaluation-only and must never enter a recovery loss.
    assert "loss_reason_missing_recovery" not in losses
    total = losses["loss_reason_selective_total"]
    gradient = torch.autograd.grad(total, latent)[0]
    assert gradient[hidden].abs().max().item() == 0.0


def test_icdor_weak_negative_is_weighted_and_unknown_remains_ignored() -> None:
    logits = torch.full((1, 2), 2.0, requires_grad=True)
    zeros = torch.zeros_like(logits)
    unknown = torch.zeros_like(logits, dtype=torch.bool)
    weak = torch.tensor([[1, 0]], dtype=torch.bool)
    losses = factor_presence_visibility_losses(logits, logits.clone(), zeros, zeros, unknown, unknown, weak)
    losses["loss_factor_total"].backward()
    assert losses["loss_factor_weak_negative"] > 0
    assert logits.grad[0, 0] > 0
    assert logits.grad[0, 1] == 0


def test_icdor_multiview_and_prototype_losses_are_real_and_finite() -> None:
    first = torch.rand(2, 3)
    second = first.clone()
    masks = torch.rand(2, 3, 4, 5)
    view = factor_view_consistency_loss(first, second, first, second, masks, masks)
    assert view["loss_factor_view_total"].item() == 0.0
    weights = torch.tensor([[[0.9, 0.1], [1.0, 0.0], [0.5, 0.5]]])
    prototypes = torch.randn(3, 2, 8)
    valid = torch.tensor([[1, 1], [1, 0], [1, 1]], dtype=torch.bool)
    regularization = factor_prototype_regularization(weights, prototypes, valid, torch.full((3,), 0.1))
    assert all(torch.isfinite(value) for value in regularization.values())
    assert regularization["loss_factor_prototype_occupancy"] > 0
