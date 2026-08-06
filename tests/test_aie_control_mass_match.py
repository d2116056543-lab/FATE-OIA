import torch

from fate_oia.utils.aie_counterfactual import matched_control_mask


def test_control_matches_selected_support_count():
    selected = torch.zeros(200); selected[:32] = 1
    control, valid, _ = matched_control_mask(selected, torch.ones(200), seed=2)
    assert valid and int(control.sum()) == int(selected.sum())

