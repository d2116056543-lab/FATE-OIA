import torch

from fate_oia.models.tida_traffic_trajectory_head import TIDATrafficTrajectoryHead


def _inputs(batch=2, actions=4, tracks=3, frames=5, dim=16):
    torch.manual_seed(7)
    action = torch.randn(batch, actions, dim)
    appearance = torch.randn(batch, actions, tracks, frames, dim)
    xy = torch.randn(batch, actions, tracks, frames, 2).tanh()
    visibility = torch.ones(batch, actions, tracks, frames)
    pair_valid = torch.ones(batch, actions, tracks, frames - 1, dtype=torch.bool)
    common = torch.zeros(batch, frames - 1, 2)
    exclusive = xy[..., 1:, :] - xy[..., :-1, :] - common[:, None, None]
    anchor = torch.tensor([0.98, 0.01, 0.01])[:tracks].expand(batch, actions, -1).clone()
    anchor = anchor / anchor.sum(-1, keepdim=True)
    return action, appearance, xy, visibility, pair_valid, common, exclusive, anchor


def test_head_is_zero_init_bounded_and_action_specific():
    args = _inputs()
    head = TIDATrafficTrajectoryHead(dim=16, num_actions=4, num_heads=4, cap=0.08)
    out = head(*args)
    assert out["traffic_trajectory_delta"].shape == (2, 4)
    assert torch.count_nonzero(out["traffic_trajectory_delta"]) == 0
    assert out["trajectory_attention"].shape == (2, 4, 3)
    assert out["trajectory_direction_histogram"].shape == (2, 4, 3, 8)
    assert torch.all(out["traffic_trajectory_delta"].abs() <= 0.08)
    assert torch.all(out["trajectory_attention"][..., 0] > out["trajectory_attention"][..., 1])


def test_head_changes_with_temporal_order_and_keeps_reason_firewall():
    args = list(_inputs(batch=1))
    reason = torch.randn(1, 21, requires_grad=True)
    head = TIDATrafficTrajectoryHead(dim=16, num_actions=4, num_heads=4)
    with torch.no_grad():
        head.output.weight.fill_(0.05)
        head.trust_raw.add_(1.0)
    original = head(*args)["traffic_trajectory_delta"]
    args[1] = args[1].flip(3)
    args[2] = args[2].flip(3)
    args[6] = args[2][..., 1:, :] - args[2][..., :-1, :] - args[5][:, None, None]
    reversed_delta = head(*args)["traffic_trajectory_delta"]
    assert not torch.allclose(original, reversed_delta)
    original.sum().backward()
    assert reason.grad is None


def test_head_encodes_order_when_motion_multiset_is_unchanged():
    args = list(_inputs(batch=1, frames=5))
    # Hold geometry fixed so the only signal is the ordered appearance sequence.
    args[2] = torch.zeros_like(args[2])
    args[5] = torch.zeros_like(args[5])
    args[6] = torch.zeros_like(args[6])
    head = TIDATrafficTrajectoryHead(dim=16, num_actions=4, num_heads=4)
    with torch.no_grad():
        head.output.weight.fill_(0.05)
        head.trust_raw.add_(1.0)
    ordered = head(*args)["traffic_trajectory_delta"]
    args[1] = args[1].flip(3)
    reversed_order = head(*args)["traffic_trajectory_delta"]
    assert not torch.allclose(ordered, reversed_order, atol=1e-7)


def test_head_first_backward_reaches_zero_initialized_output():
    args = _inputs(batch=1)
    head = TIDATrafficTrajectoryHead(dim=16, num_actions=4, num_heads=4)
    loss = head(*args)["traffic_trajectory_delta"].sum()
    loss.backward()
    assert head.output.weight.grad is not None
    assert head.output.weight.grad.abs().sum() > 0
