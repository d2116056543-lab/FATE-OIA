from __future__ import annotations

import torch

from fate_oia.utils.acpr_pair_budget import apply_pair_budget, pair_budget_ratio


def test_pair_budget_caps_weighted_pair_loss():
    used, stats = apply_pair_budget(torch.tensor(0.50), torch.tensor(0.30), epoch=4)
    assert pair_budget_ratio(4) == 0.20
    assert float(used) <= 0.060001
    assert stats["pair_budget_active"] is True


def test_pair_budget_late_ratio_is_lower():
    used, _ = apply_pair_budget(torch.tensor(0.50), torch.tensor(0.30), epoch=9)
    assert float(used) <= 0.030001

