import torch

from fate_oia.models.tida_action_reader import TIDAActionReader
from fate_oia.models.tida_reason_reader import TIDAReasonReader


def test_action_reader_routes_transition_family_and_preserves_zero_scale_fallback():
    module = TIDAActionReader(dim=8, num_actions=4, num_predicates=3)
    output = module(
        torch.randn(2, 4, 8),
        torch.randn(2, 3, 8),
        torch.randn(2, 4, 8),
        torch.ones(2, 7),
        temporal_scale=0.0,
        transition_state=torch.randn(2, 3, 8),
        transition_reliability=torch.ones(2, 3),
    )
    assert output["action_route"].shape == (2, 4, 11)
    assert output["action_flow_route_mass"].shape == (2, 4)
    assert torch.all(output["action_flow_route_mass"] > 0)
    assert torch.count_nonzero(output["action_temporal_delta"]) == 0


def test_reliable_flow_route_gives_action_flow_output_weight_gradient():
    module = TIDAActionReader(dim=8, num_actions=4, num_predicates=3)
    output = module(
        torch.randn(2, 4, 8), torch.randn(2, 3, 8), torch.randn(2, 4, 8),
        torch.ones(2, 7), temporal_scale=1.0,
        transition_state=torch.randn(2, 3, 8), transition_reliability=torch.ones(2, 3),
    )
    output["action_temporal_delta"].sum().backward()
    assert module.flow_output_weight.grad is not None
    assert module.flow_output_weight.grad.abs().sum() > 0


def test_reason_flow_keeps_transition_and_action_inputs_behind_detach_firewall():
    module = TIDAReasonReader(dim=8, num_reasons=5)
    reason_nodes = torch.randn(2, 5, 8, requires_grad=True)
    predicate = torch.randn(2, 3, 8, requires_grad=True)
    action = torch.randn(2, 4, 8, requires_grad=True)
    transition = torch.randn(2, 3, 8, requires_grad=True)
    output = module(
        reason_nodes,
        predicate,
        action,
        torch.ones(2, 7),
        temporal_scale=1.0,
        transition_state=transition,
        transition_reliability=torch.ones(2, 3),
    )
    output["reason_temporal_delta"].sum().backward()
    assert output["reason_flow_route_mass"].shape == (2, 5)
    assert reason_nodes.grad is not None and reason_nodes.grad.abs().sum() > 0
    assert predicate.grad is None
    assert action.grad is None
    assert transition.grad is None
    assert output["reason_flow_route_mass"].min() > 0
    assert module.flow_value.weight.grad is not None and module.flow_value.weight.grad.abs().sum() > 0
