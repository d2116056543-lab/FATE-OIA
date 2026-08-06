import torch

from fate_oia.losses.aie_losses import counterfactual_necessity_loss


def test_necessity_loss_prefers_selected_over_control_drop():
    valid = torch.ones(1)
    good = counterfactual_necessity_loss(torch.tensor([1.0]), torch.tensor([0.0]), valid)
    bad = counterfactual_necessity_loss(torch.tensor([0.0]), torch.tensor([1.0]), valid)
    assert good < bad

