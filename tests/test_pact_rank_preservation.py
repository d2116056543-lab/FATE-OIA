import torch

from fate_oia.losses.pact_rank_losses import action_rank_trust_region


def test_rank_trust_region_distinguishes_repairs_from_inversions():
    primary_pos, primary_neg = torch.tensor([1.0, 0.5]), torch.tensor([-1.0, -0.5])
    unchanged = action_rank_trust_region(primary_pos, primary_neg, primary_pos, primary_neg, 0.1, 0.9)
    inverted = action_rank_trust_region(primary_neg, primary_pos, primary_pos, primary_neg, 0.1, 0.9)
    assert unchanged["preserve_loss"] < inverted["preserve_loss"]
    assert unchanged["new_pair_inversion_rate"] < inverted["new_pair_inversion_rate"]
