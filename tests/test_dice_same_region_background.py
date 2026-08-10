import torch

from fate_oia.utils.dice_counterfactual import hard_region_topk, mass_matched_control


def test_control_preserves_selected_mass_and_stays_in_region():
    selected = torch.tensor([[0., .2, .3, 0.]])
    region = torch.tensor([0., 1., 1., 0.])
    control = mass_matched_control(selected, region, 1)
    assert torch.allclose(control.sum(-1), selected.sum(-1))
    assert control[0, 0] == 0 and control[0, 3] == 0


def test_deletion_mask_has_real_topk_support_inside_region():
    probability=torch.arange(10,dtype=torch.float32)
    region=torch.tensor([0,0,1,1,1,1,1,1,0,0],dtype=torch.float32)
    mask=hard_region_topk(probability,region,4)
    assert mask.sum()==4
    assert not bool((mask*(1-region)).any())
