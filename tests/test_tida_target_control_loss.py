import torch

from fate_oia.losses.tida_losses import target_token_control_margin_loss


def test_control_margin_rewards_label_correct_ordered_delta():
    base = torch.zeros(2, 2)
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    helpful = torch.tensor([[0.5, -0.5], [0.4, -0.4]])
    harmful = -helpful
    zero = torch.zeros_like(base)

    good = target_token_control_margin_loss(
        base, helpful, harmful, zero, target, margin=0.01
    )
    bad = target_token_control_margin_loss(
        base, harmful, helpful, zero, target, margin=0.01
    )

    assert good == 0
    assert bad > 0.1


def test_control_margin_ignores_unobserved_reason_without_contradiction():
    base = torch.zeros(1, 2)
    target = torch.tensor([[1.0, 0.0]])
    ordered = torch.tensor([[0.5, 10.0]])
    control = torch.tensor([[-0.5, -10.0]])
    contradiction = torch.zeros_like(target)

    loss = target_token_control_margin_loss(
        base,
        ordered,
        control,
        control,
        target,
        negative_weight=contradiction,
        margin=0.01,
    )

    assert loss == 0
