import torch

from fate_oia.models.mosaic_factor_seeded_rereader import MOSAICFactorSeededRereader


def test_typed_coordinates_change_target_output():
    module = MOSAICFactorSeededRereader(dim=8, target_count=2, grid_hw=(8, 8))
    high = torch.randn(1, 8, 8, 8)
    queries = torch.randn(1, 2, 8)
    coords = torch.zeros(1, 1, 1, 1, 2, 2)
    coords[..., 0, :] = -0.7
    coords[..., 1, :] = 0.7
    sampled = torch.randn(1, 1, 1, 1, 2, 8)
    attention = torch.ones(1, 1, 1, 1, 2)
    weights = torch.ones(1, 1, 2)
    first = module(high, queries, coords, sampled, attention, weights)["target_nodes"]
    moved = coords.clone()
    # Move both equally weighted samples so the factor coordinate centroid
    # changes; swapping them would leave the rereader input unchanged.
    moved[..., 0, :] = 0.7
    moved[..., 1, :] = 0.7
    second = module(high, queries, moved, sampled, attention, weights)["target_nodes"]
    assert not torch.allclose(first, second)
