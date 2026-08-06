import torch

from fate_oia.utils.aie_counterfactual import matched_control_mask


def test_control_fails_closed_or_meets_overlap_bound():
    selected = torch.zeros(500); selected[:64] = 1
    _, valid, overlap = matched_control_mask(selected, torch.ones(500), seed=123, max_overlap=0.20)
    assert (valid and overlap <= 0.20) or not valid

