import torch

from fate_oia.models.tida_action_motion_cross_attention import TIDAActionMotionCrossAttention


def _inputs(batch: int = 2, frames: int = 6, dim: int = 32):
    action_nodes = torch.randn(batch, 4, dim)
    history = torch.randn(batch, frames, 4, dim)
    timestamps = torch.linspace(-2.0, -0.1, frames).repeat(batch, 1)
    valid = torch.ones(batch, frames, dtype=torch.bool)
    logits = torch.randn(batch, 4)
    return action_nodes, history, timestamps, valid, logits


def test_zero_initialization_is_exact_noop_but_output_projection_gets_gradient():
    module = TIDAActionMotionCrossAttention(dim=32, num_actions=4, cap=0.15)
    result = module(*_inputs())
    assert torch.equal(result["traffic_action_delta"], torch.zeros_like(result["traffic_action_delta"]))
    result["traffic_action_delta"].sum().backward()
    assert module.output.weight.grad is not None and torch.isfinite(module.output.weight.grad).all()


def test_invalid_history_is_exact_zero_and_attention_is_well_formed():
    module = TIDAActionMotionCrossAttention(dim=32, num_actions=4, cap=0.15)
    action_nodes, history, timestamps, valid, logits = _inputs()
    result = module(action_nodes, history, timestamps, torch.zeros_like(valid), logits)
    assert torch.equal(result["traffic_action_delta"], torch.zeros_like(result["traffic_action_delta"]))
    assert torch.equal(result["traffic_motion_energy"], torch.zeros_like(result["traffic_motion_energy"]))
    assert torch.isfinite(result["traffic_action_attention"]).all()


def test_motion_direction_and_time_reversal_change_the_learned_context():
    module = TIDAActionMotionCrossAttention(dim=32, num_actions=4, cap=0.15)
    action_nodes, history, timestamps, valid, logits = _inputs(batch=1)
    ordered = module(action_nodes, history, timestamps, valid, logits)
    reversed_result = module(action_nodes, history.flip(1), timestamps, valid, logits)
    assert not torch.allclose(ordered["traffic_action_context"], reversed_result["traffic_action_context"])
    assert ordered["traffic_action_attention"].shape == (1, 4, 20)


def test_action_targets_have_distinct_same_action_attention_diagnostic():
    module = TIDAActionMotionCrossAttention(dim=32, num_actions=4, cap=0.15)
    result = module(*_inputs(batch=1))
    assert result["traffic_same_action_mass"].shape == (1, 4)
    assert torch.all((result["traffic_same_action_mass"] >= 0) & (result["traffic_same_action_mass"] <= 1))


def test_sparse_patch_correspondence_recovers_directional_displacement():
    module = TIDAActionMotionCrossAttention(dim=32, num_actions=4, cap=0.15)
    action_nodes, history, timestamps, valid, logits = _inputs(batch=1, frames=2)
    patch = torch.randn(1, 2, 4, 2, 32)
    patch[:, 1] = patch[:, 0]
    xy = torch.zeros(1, 2, 4, 2, 2)
    xy[:, 0, :, 0, 0] = -0.8
    xy[:, 0, :, 1, 0] = -0.4
    xy[:, 1, :, 0, 0] = 0.4
    xy[:, 1, :, 1, 0] = 0.8
    weight = torch.full((1, 2, 4, 2), 0.5)
    result = module(
        action_nodes, history, timestamps, valid, logits,
        patch_tokens=patch, patch_xy=xy, patch_weight=weight,
    )
    assert result["traffic_patch_displacement"][..., 0].mean() > 0.5
    assert result["traffic_patch_match_confidence"].mean() > 0.5


def test_sparse_patch_correspondence_separates_common_and_action_specific_motion():
    module = TIDAActionMotionCrossAttention(dim=8, num_actions=2, cap=0.05)
    action_nodes = torch.randn(1, 2, 8)
    history = torch.randn(1, 2, 2, 8)
    timestamps = torch.tensor([[0.0, 1.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    base_logits = torch.zeros(1, 2)
    patch_tokens = torch.zeros(1, 2, 2, 2, 8)
    patch_tokens[0, 0, 0, 0, 0] = 1
    patch_tokens[0, 0, 0, 1, 1] = 1
    patch_tokens[0, 0, 1, 0, 2] = 1
    patch_tokens[0, 0, 1, 1, 3] = 1
    patch_tokens[:, 1] = patch_tokens[:, 0]
    patch_xy = torch.tensor(
        [[[[[-0.5, 0.0], [0.5, 0.0]], [[-0.5, 0.0], [0.5, 0.0]]],
          [[[-0.3, 0.0], [0.7, 0.0]], [[0.1, 0.0], [1.1, 0.0]]]]]
    )
    patch_weight = torch.full((1, 2, 2, 2), 0.5)

    result = module(
        action_nodes,
        history,
        timestamps,
        valid,
        base_logits,
        patch_tokens=patch_tokens,
        patch_xy=patch_xy,
        patch_weight=patch_weight,
    )

    raw = result["traffic_patch_displacement"][0, 0, :, 0]
    exclusive = result["traffic_patch_exclusive_displacement"][0, 0, :, 0]
    torch.testing.assert_close(raw, torch.tensor([0.2, 0.6]), atol=1e-4, rtol=0)
    torch.testing.assert_close(exclusive, torch.tensor([-0.2, 0.2]), atol=1e-4, rtol=0)
    torch.testing.assert_close(exclusive.mean(), torch.tensor(0.0), atol=1e-5, rtol=0)
    assert result["traffic_patch_exclusive_motion_energy"].shape == (1, 1, 2)
    assert result["traffic_patch_effective_motion"].min() > 0
