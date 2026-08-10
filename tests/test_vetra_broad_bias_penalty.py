import torch
from fate_oia.losses.vetra_losses import correction_bias_energy_loss


def test_broad_constant_bias_is_penalized_more_than_centered_delta():
    constant = torch.ones(8,4) * .1
    centered = torch.tensor([[-.1,.1,-.1,.1],[.1,-.1,.1,-.1]]).repeat(4,1)
    assert correction_bias_energy_loss(constant)[0] > correction_bias_energy_loss(centered)[0]
