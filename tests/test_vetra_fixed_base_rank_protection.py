import torch
from fate_oia.losses.vetra_losses import fixed_base_rank_protection_loss


def test_fixed_base_trust_region_penalizes_inversion_only():
    target = torch.tensor([[1.,0.],[0.,1.]])
    base = torch.tensor([[1.,-1.],[-1.,1.]])
    safe = fixed_base_rank_protection_loss(base.clone(), base, target)
    inverted = fixed_base_rank_protection_loss(-base, base, target)
    assert safe == 0 and inverted > safe
