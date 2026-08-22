import torch

from fate_oia.models.tida_temporal_utility import TIDAConditionalTemporalUtility


def _module():
    return TIDAConditionalTemporalUtility(max_budget=0.60, min_budget=0.02)


def test_uncertain_image_and_strong_motion_receive_larger_budget():
    module = _module()
    shape = (2, 4)
    common = dict(
        motion_salience=torch.ones(shape),
        transition_consistency=torch.ones(shape),
        compatibility=torch.zeros(shape),
        history_available=torch.ones(2, dtype=torch.bool),
    )
    uncertain = module(torch.zeros(shape), **common)
    certain = module(torch.full(shape, 8.0), **common)

    assert torch.all(uncertain["budget"] > certain["budget"])


def test_no_history_has_exact_zero_budget():
    shape = (2, 4)
    output = _module()(
        torch.zeros(shape),
        torch.ones(shape),
        torch.ones(shape),
        torch.zeros(shape),
        torch.zeros(2, dtype=torch.bool),
    )

    assert torch.count_nonzero(output["budget"]) == 0
    assert torch.count_nonzero(output["need"]) == 0


def test_budget_is_target_specific_bounded_and_compatibility_trainable():
    module = _module()
    compatibility = torch.tensor([[-4.0, -1.0, 1.0, 4.0]], requires_grad=True)
    output = module(
        torch.zeros(1, 4),
        torch.ones(1, 4),
        torch.ones(1, 4),
        compatibility,
        torch.ones(1, dtype=torch.bool),
    )

    assert output["budget"].shape == (1, 4)
    assert output["budget"].min() >= 0.02
    assert output["budget"].max() <= 0.60
    assert torch.all(output["budget"][:, 1:] > output["budget"][:, :-1])
    output["budget"].sum().backward()
    assert compatibility.grad is not None and compatibility.grad.abs().sum() > 0


def test_motion_salience_is_normalized_without_binary_saturation():
    output = _module()(
        torch.zeros(1, 3),
        torch.tensor([[0.1, 1.0, 10.0]]),
        torch.ones(1, 3),
        torch.zeros(1, 3),
        torch.ones(1, dtype=torch.bool),
    )

    normalized = output["motion_weight"]
    assert torch.all(normalized[:, 1:] > normalized[:, :-1])
    assert normalized.min() > 0
    assert normalized.max() < 1
