import torch

from fate_oia.utils.acpr_pair_budget import apply_pair_budget, pair_budget_ratio


def test_pair_budget_caps_pair_loss_against_action_reason_main_only():
    pair_raw = torch.tensor(10.0)
    action = torch.tensor(1.0)
    reason = torch.tensor(1.0)
    used, stats = apply_pair_budget(pair_raw, action, reason, epoch=8)

    assert torch.allclose(used, torch.tensor(0.2))
    assert stats["pair_budget_ratio"] == 0.10
    assert pair_budget_ratio(2) == 0.0
    assert pair_budget_ratio(4) == 0.20
    assert pair_budget_ratio(8) == 0.10
