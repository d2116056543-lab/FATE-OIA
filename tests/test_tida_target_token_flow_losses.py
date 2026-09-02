import torch

from fate_oia.engine.train_tida_oia import target_token_phase_loss_weights
from fate_oia.losses.tida_losses import target_conditioned_pu_correction_loss


def test_first_two_epochs_are_predictive_only_then_enable_supervision():
    config = {
        "loss": {
            "target_token_action_prediction": 0.10,
            "target_token_action_order": 0.05,
            "target_token_action_aux": 0.80,
            "target_token_action_rank": 0.20,
            "target_token_action_utility": 0.05,
            "target_token_action_no_harm": 0.30,
            "target_token_action_delta": 0.002,
            "target_token_reason_prediction": 0.10,
            "target_token_reason_order": 0.05,
            "target_token_reason_aux": 0.70,
            "target_token_reason_rank": 0.08,
            "target_token_reason_utility": 0.04,
            "target_token_reason_no_harm": 0.25,
            "target_token_reason_delta": 0.002,
        },
        "training": {
            "target_token_predictive_epochs": 2,
            "target_token_predictive_loss": {
                "target_token_action_prediction": 0.50,
                "target_token_action_order": 0.15,
                "target_token_reason_prediction": 0.40,
                "target_token_reason_order": 0.12,
            },
        },
    }

    early = target_token_phase_loss_weights(config, epoch=1)
    late = target_token_phase_loss_weights(config, epoch=2)

    assert early["target_token_action_prediction"] == 0.50
    assert early["target_token_reason_order"] == 0.12
    assert early["target_token_action_aux"] == 0.0
    assert early["target_token_reason_utility"] == 0.0
    assert late == config["loss"]


def test_reason_unknown_rows_are_not_used_as_hard_negatives():
    base = torch.tensor([[0.0], [0.0]])
    target = torch.tensor([[1.0], [0.0]])
    contradiction = torch.zeros_like(target)
    motion = torch.ones_like(target)
    delta_a = torch.tensor([[0.1], [-5.0]], requires_grad=True)
    delta_b = torch.tensor([[0.1], [5.0]], requires_grad=True)

    loss_a = target_conditioned_pu_correction_loss(
        base, delta_a, target, motion, contradiction
    )
    loss_b = target_conditioned_pu_correction_loss(
        base, delta_b, target, motion, contradiction
    )

    assert torch.allclose(loss_a, loss_b)
