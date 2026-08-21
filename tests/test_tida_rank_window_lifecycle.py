import torch

from fate_oia.engine.train_tida_oia import append_rank_window, clear_rank_window, rank_window_reference


def test_rank_window_is_detached_and_cleared_at_optimizer_boundary():
    window = {}
    action_logits = torch.randn(2, 4, requires_grad=True)
    reason_logits = torch.randn(2, 21, requires_grad=True)
    action_target = torch.zeros(2, 4)
    reason_target = torch.zeros(2, 21)
    reason_weight = torch.full((2, 21), 0.2)

    append_rank_window(
        window,
        action_logits,
        action_target,
        reason_logits,
        reason_target,
        reason_weight,
    )
    reference = rank_window_reference(window)

    assert reference is not None
    assert reference["action_logits"].shape == (2, 4)
    assert reference["reason_logits"].shape == (2, 21)
    assert not any(value.requires_grad for value in reference.values())

    clear_rank_window(window)
    assert rank_window_reference(window) is None
