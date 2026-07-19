from __future__ import annotations

import torch

from fate_oia.losses.mosaic_icdor_reason_losses import selective_observation_losses


def test_pu_gate_only_disables_recovery_terms_not_observed_annotation_loss() -> None:
    """Observed labels remain supervised even when no label is admitted to PU recovery."""
    logits = torch.tensor([[0.4] * 21], requires_grad=True)
    observation_logits = torch.full_like(logits, 0.7, requires_grad=True)
    target = torch.zeros_like(logits)
    target[:, 0] = 1.0
    output = selective_observation_losses(
        logits,
        target,
        torch.full_like(logits, 0.8),
        torch.full_like(logits, 0.2),
        reason_observation_logits=observation_logits,
        reason_propensity=torch.full_like(logits, 0.1),
        factor_route_support=torch.full_like(logits, 0.3),
        escape_weight=torch.full_like(logits, 0.1),
        pu_gate=torch.zeros(21, dtype=torch.bool),
    )
    assert output["loss_reason_observation_nll"].item() > 0.0
    assert output["loss_reason_posterior_bce"].item() == 0.0
    output["loss_reason_observation_nll"].backward()
    assert observation_logits.grad is not None and torch.isfinite(observation_logits.grad).all()
