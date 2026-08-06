import torch


def test_union_mask_is_probability_union_not_sum():
    masks = torch.tensor([[0.8, 0.0], [0.7, 0.5]])
    union = 1 - torch.prod(1 - masks, dim=0)
    torch.testing.assert_close(union, torch.tensor([0.94, 0.50]))
    assert bool((union <= 1).all())

