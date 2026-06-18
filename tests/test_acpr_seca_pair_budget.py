import torch

from fate_oia.utils.acpr_pair_budget import apply_pair_budget


def test_pair_budget_caps_weighted_pair_loss():
    main = torch.tensor(2.0)
    used, stats = apply_pair_budget(torch.tensor(10.0), torch.tensor(10.0), pair_logit_weight=1.0, pair_embed_weight=1.0, main_loss=main, pair_budget_ratio=0.25)
    assert used <= 0.5 + 1e-6
    assert stats["pair_budget_active"] is True
    assert stats["pair_to_main_ratio_used"] <= 0.25 + 1e-6
