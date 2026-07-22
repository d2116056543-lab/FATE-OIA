import torch

from fate_oia.losses.precise_intervention_losses import matched_control_is_valid, target_specific_intervention_loss


def test_selected_effect_and_wrong_target_have_required_loss_direction():
    base = torch.zeros(2, 4)
    target = torch.ones(2, 4)
    good = target_specific_intervention_loss(torch.full((2, 4), 0.5), torch.full((2, 4), 0.1), torch.full((2, 4), 0.1), base, base, target)["loss_intervention"]
    bad = target_specific_intervention_loss(torch.full((2, 4), 0.1), torch.full((2, 4), 0.5), torch.full((2, 4), 0.6), base, base, target)["loss_intervention"]
    assert good < bad


def test_matched_control_enforces_mass_and_nonoverlap():
    selected = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    control = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    valid = matched_control_is_valid(selected, control, torch.tensor([True]), torch.tensor([True]), torch.tensor([True]))
    assert valid.item() is True
