import torch

from fate_oia.models.tida_traffic_trajectories import TIDATrafficTrajectoryBuilder


def _moving_patches():
    # Two stable identities move right with distinct appearances.
    tokens = torch.zeros(1, 3, 1, 2, 4)
    tokens[..., 0, 0] = 1.0
    tokens[..., 1, 1] = 1.0
    xy = torch.tensor(
        [[[[[-0.8, 0.0], [0.2, 0.2]]], [[[-0.6, 0.0], [0.4, 0.2]]], [[[-0.4, 0.0], [0.6, 0.2]]]]]
    )
    weights = torch.full((1, 3, 1, 2), 0.5)
    valid = torch.ones(1, 3, dtype=torch.bool)
    return tokens, xy, weights, valid


def test_builder_preserves_final_anchor_identity_and_motion():
    tokens, xy, weights, valid = _moving_patches()
    out = TIDATrafficTrajectoryBuilder(temperature=0.03)(tokens, xy, weights, valid)
    assert out["trajectory_xy"].shape == (1, 1, 2, 3, 2)
    assert torch.allclose(out["trajectory_xy"][0, 0, :, -1], xy[0, -1, 0])
    displacement = out["trajectory_displacement"][0, 0]
    assert torch.allclose(displacement[..., 0], torch.full((2, 2), 0.2), atol=1e-3)
    assert out["trajectory_cycle_confidence"].mean() > 0.95


def test_builder_masks_invalid_pairs_and_returns_finite_gradients():
    tokens, xy, weights, valid = _moving_patches()
    tokens.requires_grad_(True)
    valid[:, 1] = False
    out = TIDATrafficTrajectoryBuilder()(tokens, xy, weights, valid)
    assert not out["trajectory_pair_valid"].any()
    assert torch.count_nonzero(out["trajectory_displacement"]) == 0
    loss = out["trajectory_appearance"].square().mean()
    loss.backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()


def test_builder_separates_common_camera_motion_from_exclusive_motion():
    tokens, xy, weights, valid = _moving_patches()
    # Add a second action whose points share camera motion but have extra vertical motion.
    tokens = tokens.expand(-1, -1, 2, -1, -1).clone()
    xy = xy.expand(-1, -1, 2, -1, -1).clone()
    xy[:, 1:, 1, :, 1] += torch.tensor([0.1, 0.2]).view(1, 2, 1)
    weights = weights.expand(-1, -1, 2, -1).clone()
    out = TIDATrafficTrajectoryBuilder(temperature=0.03)(tokens, xy, weights, valid)
    common = out["trajectory_common_displacement"]
    exclusive = out["trajectory_exclusive_displacement"]
    assert torch.allclose(common[..., 0], torch.full_like(common[..., 0], 0.2), atol=1e-3)
    assert exclusive[:, 0, :, :, 1].mean() < 0
    assert exclusive[:, 1, :, :, 1].mean() > 0
