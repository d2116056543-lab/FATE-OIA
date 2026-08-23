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
