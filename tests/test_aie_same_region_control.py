import torch

from fate_oia.utils.aie_counterfactual import matched_control_mask


def test_control_is_drawn_from_requested_region():
    selected = torch.zeros(100); selected[:10] = 1
    region = torch.zeros(100); region[:80] = 1
    control, valid, _ = matched_control_mask(selected, region, seed=7)
    assert valid and float((control * (1 - region)).sum()) == 0

