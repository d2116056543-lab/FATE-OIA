import torch

from fate_oia.models.tida_oia_model import TIDAOIAModel


def test_left_right_name_swap_is_involutive():
    names = ("vehicle_left", "vehicle_right", "road_clear")
    assert tuple(TIDAOIAModel._swap_lr_name(TIDAOIAModel._swap_lr_name(name)) for name in names) == names


def test_action_flip_permutation_is_involutive():
    values = torch.arange(4)
    permutation = torch.tensor([0, 1, 3, 2])
    assert torch.equal(values.index_select(0, permutation).index_select(0, permutation), values)


def test_ego_left_right_channel_swap_is_involutive():
    ego = torch.randn(45, 80, 8)
    flipped = ego.flip(1).clone()
    flipped[..., 0] = 1.0 - flipped[..., 0]
    flipped[..., [3, 4]] = flipped[..., [4, 3]]
    restored = flipped.flip(1).clone()
    restored[..., 0] = 1.0 - restored[..., 0]
    restored[..., [3, 4]] = restored[..., [4, 3]]
    torch.testing.assert_close(restored, ego)
