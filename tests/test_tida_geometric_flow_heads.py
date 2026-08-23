import torch

from fate_oia.models.tida_geometric_flow import TIDAGeometricFlowDecisionHeads


def test_geometric_heads_start_as_exact_noop_but_receive_gradient():
    heads = TIDAGeometricFlowDecisionHeads(hidden_dim=32, num_actions=4, num_reasons=21)
    state = torch.randn(3, 32)
    output = heads(state, history_available=torch.ones(3, dtype=torch.bool))
    assert torch.equal(output["geometric_action_delta"], torch.zeros(3, 4))
    assert torch.equal(output["geometric_reason_delta"], torch.zeros(3, 21))
    loss = output["geometric_action_delta"].sum() + output["geometric_reason_delta"].sum()
    loss.backward()
    assert heads.action_output.weight.grad.abs().sum() > 0
    assert heads.reason_output.weight.grad.abs().sum() > 0


def test_action_and_reason_heads_have_a_parameter_firewall():
    heads = TIDAGeometricFlowDecisionHeads(hidden_dim=32, num_actions=4, num_reasons=21)
    state = torch.randn(3, 32)
    output = heads(state, history_available=torch.ones(3, dtype=torch.bool))
    torch.autograd.grad(output["geometric_reason_delta"].sum(), list(heads.action_parameters()), allow_unused=True)
    grads = torch.autograd.grad(output["geometric_action_delta"].sum(), list(heads.reason_parameters()), allow_unused=True)
    assert all(value is None or torch.equal(value, torch.zeros_like(value)) for value in grads)


def test_unavailable_history_forces_exact_fallback():
    heads = TIDAGeometricFlowDecisionHeads(hidden_dim=16, num_actions=4, num_reasons=21)
    torch.nn.init.normal_(heads.action_output.weight)
    torch.nn.init.normal_(heads.reason_output.weight)
    output = heads(torch.randn(2, 16), history_available=torch.zeros(2, dtype=torch.bool))
    assert torch.equal(output["geometric_action_delta"], torch.zeros(2, 4))
    assert torch.equal(output["geometric_reason_delta"], torch.zeros(2, 21))
