import torch

from fate_oia.losses.tfc_losses import factor_measurement_loss
from fate_oia.models.tfc_target_credit import TFCTargetCredit


def test_factor_measurement_loss_is_finite_for_bfloat16_saturated_probs():
    factor_probs_action = torch.ones(2, 12, dtype=torch.bfloat16)
    factor_probs_reason = torch.ones(2, 12, dtype=torch.bfloat16)

    loss = factor_measurement_loss(factor_probs_action, factor_probs_reason)

    assert torch.isfinite(loss)


def test_factor_measurement_loss_penalizes_all_on_more_than_sparse_target():
    all_on = torch.ones(4, 12)
    sparse = torch.zeros(4, 12)
    sparse[:, :3] = 1.0
    healthy_rho = torch.full((4, 12), 0.55)

    all_on_loss = factor_measurement_loss(all_on, all_on, all_on, all_on)
    sparse_loss = factor_measurement_loss(sparse, sparse, healthy_rho, healthy_rho)

    assert all_on_loss > sparse_loss + 0.1


def test_target_credit_confidence_does_not_reward_all_on_factor_activation():
    module = TFCTargetCredit(num_factors=4, action_dim=1, reason_dim=1, dim=4)
    factor_features = torch.randn(1, 4, 4)
    compatibility = {
        "factor_to_action_support": torch.ones(4, 1),
        "factor_to_action_inhibit": torch.zeros(4, 1),
        "factor_to_reason_support": torch.ones(4, 1),
        "factor_to_reason_inhibit": torch.zeros(4, 1),
    }
    one_active = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    all_active = torch.ones(1, 4)

    one = module(one_active, one_active, factor_features, compatibility)
    all_on = module(all_active, all_active, factor_features, compatibility)

    assert all_on["credit_confidence_action"].max() <= one["credit_confidence_action"].max() * 1.25
    assert all_on["credit_confidence_reason"].max() <= one["credit_confidence_reason"].max() * 1.25
