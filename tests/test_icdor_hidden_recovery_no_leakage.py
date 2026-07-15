from __future__ import annotations

import torch
import pytest

from fate_oia.engine.mosaic_icdor_hidden_recovery_audit import (
    audit_hidden_recovery,
    audit_hidden_recovery_scores,
    build_hidden_mask,
)
from fate_oia.losses.mosaic_icdor_reason_losses import reason_observed_losses, selective_observation_losses


def test_hidden_positive_label_has_no_gradient_path_through_any_training_loss() -> None:
    hidden_source = torch.tensor([[1.0] + [0.0] * 20], requires_grad=True)
    hidden_mask = torch.zeros(1, 21, dtype=torch.bool)
    hidden_mask[0, 0] = True
    observed = hidden_source.masked_fill(hidden_mask, 0.0)
    valid = ~hidden_mask
    visual = torch.zeros(1, 21, requires_grad=True)
    mixed = torch.zeros(1, 21, requires_grad=True)
    latent = torch.zeros(1, 21, requires_grad=True)
    probability = torch.full((1, 21), 0.5, requires_grad=True)
    posterior = torch.full((1, 21), 0.5)

    direct = reason_observed_losses(visual, mixed, observed, observed_valid_mask=valid)
    selective = selective_observation_losses(
        latent, observed, probability, posterior,
        reason_propensity=torch.full((1, 21), 0.5),
        factor_route_support=torch.zeros(1, 21),
        escape_weight=torch.zeros(1, 21),
        observed_valid_mask=valid,
    )
    assert "loss_reason_missing_recovery" not in selective
    total = direct["loss_reason_observed_total"] + selective["loss_reason_selective_total"]
    grad = torch.autograd.grad(total, hidden_source, allow_unused=True)[0]
    assert grad is None or grad[hidden_mask].abs().max().item() == 0.0


def test_hidden_recovery_ap_compares_hidden_positives_with_real_observed_zeros() -> None:
    hidden = torch.tensor([[1, 0], [0, 1], [0, 0]], dtype=torch.bool)
    observed_after_hiding = torch.zeros(3, 2)
    posterior = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.1, 0.2]])
    zero_as_negative = torch.full((3, 2), 0.5)

    result = audit_hidden_recovery_scores(
        posterior,
        zero_as_negative,
        observed_after_hiding,
        hidden,
        mode="mcar",
        hide_fraction=0.10,
    )

    assert result["available"] is True
    assert result["hidden_positive_count"] == 2
    assert result["eligible_negative_count"] == 4
    assert result["recovery_auprc"] > result["zero_as_negative_auprc"]
    assert result["margin"] > 0.0
    assert result["hide_fraction"] == 0.10
    assert result["evaluation_only"] is True


def test_hidden_recovery_abstains_instead_of_scoring_positive_only_rows() -> None:
    hidden = torch.ones(2, 2, dtype=torch.bool)
    result = audit_hidden_recovery_scores(
        torch.full((2, 2), 0.9),
        torch.full((2, 2), 0.5),
        torch.zeros(2, 2),
        hidden,
        mode="mnar",
        hide_fraction=0.50,
    )
    assert result["available"] is False
    assert result["recovery_auprc"] is None


def test_hidden_recovery_rejects_unplanned_hide_fraction() -> None:
    with torch.no_grad():
        try:
            audit_hidden_recovery_scores(
                torch.full((2, 2), 0.5),
                torch.full((2, 2), 0.5),
                torch.zeros(2, 2),
                torch.tensor([[True, False], [False, False]]),
                mode="mcar",
                hide_fraction=0.25,
            )
        except ValueError:
            return
    raise AssertionError("IC-DOR hidden-recovery audit accepted an unplanned hide fraction")


def test_all_public_hidden_recovery_entries_reject_unplanned_fraction() -> None:
    labels = torch.tensor([[1.0, 0.0]])
    with pytest.raises(ValueError):
        build_hidden_mask(labels, mode="mcar", hide_fraction=0.25)
    with pytest.raises(ValueError):
        audit_hidden_recovery(torch.zeros_like(labels), labels, hide_fraction=0.25)
