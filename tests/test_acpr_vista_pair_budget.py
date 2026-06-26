from __future__ import annotations

import torch

from fate_oia.engine.train_acpr_oia import pair_budget_main_reference
from fate_oia.utils.acpr_pair_budget import apply_pair_budget, pair_budget_ratio


def test_pair_budget_caps_weighted_pair_loss():
    used, stats = apply_pair_budget(torch.tensor(0.50), torch.tensor(0.30), epoch=4)
    assert pair_budget_ratio(4) == 0.20
    assert float(used) <= 0.060001
    assert stats["pair_budget_active"] is True


def test_pair_budget_late_ratio_is_lower():
    used, _ = apply_pair_budget(torch.tensor(0.50), torch.tensor(0.30), epoch=9)
    assert float(used) <= 0.030001


def test_pair_budget_reference_excludes_non_main_auxiliary_losses():
    terms = {
        "action_direct": torch.tensor(1.0),
        "reason_partial": torch.tensor(2.0),
        "action_visual_aux": torch.tensor(3.0),
        "action_reason_aux": torch.tensor(4.0),
        "predicate_weak": torch.tensor(100.0),
        "calibration": torch.tensor(100.0),
        "action_combo_ce": torch.tensor(100.0),
    }
    weights = {
        "action_direct": 1.0,
        "reason_partial": 1.0,
        "action_visual_aux": 0.05,
        "action_reason_aux": 0.05,
        "predicate_weak": 1.0,
        "calibration": 1.0,
        "action_combo_ce": 1.0,
    }
    ref = pair_budget_main_reference(terms, weights)
    assert torch.isclose(ref, torch.tensor(3.35))
