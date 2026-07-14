from __future__ import annotations

import torch

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

