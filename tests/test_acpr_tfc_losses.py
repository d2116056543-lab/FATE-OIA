import torch

from fate_oia.losses.tfc_losses import factor_measurement_loss


def test_factor_measurement_loss_is_finite_for_bfloat16_saturated_probs():
    factor_probs_action = torch.ones(2, 12, dtype=torch.bfloat16)
    factor_probs_reason = torch.ones(2, 12, dtype=torch.bfloat16)

    loss = factor_measurement_loss(factor_probs_action, factor_probs_reason)

    assert torch.isfinite(loss)
