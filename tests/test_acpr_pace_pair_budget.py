import torch

from fate_oia.utils.acpr_pair_budget import apply_pair_budget


def test_pair_budget_caps_relative_to_main_losses():
    used, stats = apply_pair_budget(torch.tensor(0.50), torch.tensor(0.10), torch.tensor(0.20), ratio=0.25, epsilon=0.0)
    assert abs(float(used) - 0.075) < 1e-6
    assert stats["pair_budget_active"] is True
    unchanged, stats2 = apply_pair_budget(torch.tensor(0.01), torch.tensor(0.10), torch.tensor(0.20), ratio=0.25, epsilon=0.0)
    assert abs(float(unchanged) - 0.01) < 1e-6
    assert stats2["pair_budget_active"] is False
