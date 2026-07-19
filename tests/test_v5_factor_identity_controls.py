from __future__ import annotations

import torch

from fate_oia.losses.mosaic_icdor_factor_losses import (
    factor_image_identity_loss,
    factor_prior_gap_loss,
    factor_query_identity_loss,
)


def test_factor_identity_losses_use_semantic_and_image_matched_controls() -> None:
    """V5 identity losses must compare actual controls, not vector self-norms."""
    features = torch.tensor(
        [[[4.0, 0.0], [0.0, 4.0]], [[-4.0, 0.0], [0.0, -4.0]]],
        requires_grad=True,
    )
    queries = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    types = torch.tensor([0, 0])
    targets = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    known = torch.ones_like(targets, dtype=torch.bool)

    query_good = factor_query_identity_loss(features, queries, types, targets, known)
    query_bad = factor_query_identity_loss(features, queries.flip(0), types, targets, known)

    logits_good = torch.tensor([[4.0, 4.0], [-4.0, -4.0]], requires_grad=True)
    logits_bad = -logits_good.detach().clone().requires_grad_(True)
    image_good = factor_image_identity_loss(logits_good, targets, known)
    image_bad = factor_image_identity_loss(logits_bad, targets, known)

    prior = torch.zeros_like(logits_good)
    prior_good = factor_prior_gap_loss(logits_good, prior, targets, known)
    prior_bad = factor_prior_gap_loss(logits_bad, prior, targets, known)

    assert query_good < query_bad
    assert image_good < image_bad
    assert prior_good < prior_bad
    (query_good + image_good + prior_good).backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert logits_good.grad is not None and torch.isfinite(logits_good.grad).all()
