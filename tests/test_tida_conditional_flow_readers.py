import torch

from fate_oia.models.tida_action_reader import TIDAActionReader
from fate_oia.models.tida_reason_reader import TIDAReasonReader


def _transition_inputs(batch: int = 2, predicates: int = 3, dim: int = 8):
    return {
        "transition_state": torch.randn(batch, predicates, dim),
        "transition_tokens_by_scale": torch.randn(batch, predicates, 4, dim),
        "transition_reliability": torch.ones(batch, predicates),
        "motion_salience": torch.tensor([[0.1, 0.5, 2.0], [1.5, 0.2, 0.8]]),
        "transition_consistency": torch.tensor([[0.2, 0.8, 1.0], [0.9, 0.4, 0.7]]),
        "history_available": torch.tensor([True, True]),
    }


def test_action_conditional_utility_routes_typed_scales_with_target_specific_budget():
    torch.manual_seed(7)
    reader = TIDAActionReader(
        dim=8,
        num_actions=4,
        num_predicates=3,
        conditional_utility_enabled=True,
        conditional_flow_mix_cap=0.60,
    )
    output = reader(
        torch.randn(2, 4, 8),
        torch.randn(2, 3, 8),
        torch.randn(2, 4, 8),
        torch.ones(2, 7),
        temporal_scale=1.0,
        image_logits=torch.tensor([[0.0, 4.0, -4.0, 1.0], [2.0, 0.0, -1.0, 3.0]]),
        **_transition_inputs(),
    )
    assert output["action_temporal_budget"].shape == (2, 4)
    assert output["action_route"].shape == (2, 4, 20)
    assert torch.allclose(output["action_flow_route_mass"], output["action_temporal_budget"], atol=1e-6)
    assert output["action_temporal_budget"].max() <= 0.60 + 1e-7
    assert output["action_temporal_budget"].min() >= reader.flow_mix_cap - 1e-7
    assert torch.unique(output["action_temporal_budget"].round(decimals=6)).numel() > 2


def test_action_conditional_utility_history_off_is_exact_zero():
    reader = TIDAActionReader(
        dim=8, num_actions=4, num_predicates=3, conditional_utility_enabled=True
    )
    transition = _transition_inputs()
    transition["history_available"] = torch.zeros(2, dtype=torch.bool)
    output = reader(
        torch.randn(2, 4, 8), torch.randn(2, 3, 8), torch.randn(2, 4, 8),
        torch.ones(2, 7), temporal_scale=1.0, image_logits=torch.zeros(2, 4), **transition,
    )
    assert torch.count_nonzero(output["action_temporal_budget"]) == 0
    assert torch.count_nonzero(output["action_flow_route_mass"]) == 0
    assert torch.count_nonzero(output["action_temporal_delta"]) == 0


def test_reason_conditional_utility_keeps_shared_inputs_behind_firewall():
    reader = TIDAReasonReader(
        dim=8,
        num_reasons=5,
        conditional_utility_enabled=True,
        conditional_flow_mix_cap=0.50,
    )
    reasons = torch.randn(2, 5, 8, requires_grad=True)
    predicates = torch.randn(2, 3, 8, requires_grad=True)
    actions = torch.randn(2, 4, 8, requires_grad=True)
    transition = _transition_inputs()
    transition["transition_state"].requires_grad_(True)
    transition["transition_tokens_by_scale"].requires_grad_(True)
    output = reader(
        reasons, predicates, actions, torch.ones(2, 7), temporal_scale=1.0,
        image_logits=torch.zeros(2, 5), **transition,
    )
    output["reason_temporal_delta"].sum().backward()
    assert output["reason_temporal_budget"].shape == (2, 5)
    assert torch.allclose(output["reason_flow_route_mass"], output["reason_temporal_budget"], atol=1e-6)
    assert output["reason_temporal_budget"].max() <= 0.50 + 1e-7
    assert output["reason_temporal_budget"].min() >= reader.flow_mix_cap - 1e-7
    assert reasons.grad is not None and reasons.grad.abs().sum() > 0
    assert predicates.grad is None
    assert actions.grad is None
    assert transition["transition_state"].grad is None
    assert transition["transition_tokens_by_scale"].grad is None
