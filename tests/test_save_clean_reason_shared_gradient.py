import pytest
import torch
from torch import nn

try:
    from fate_oia.models.save_reason_decoder import (
        SAVECleanReasonRoute,
        SAVEReasonDecoder,
    )
except ImportError:
    SAVECleanReasonRoute = None
    SAVEReasonDecoder = None


def test_clean_reason_loss_reaches_shared_core_and_reason_to_action_bridge() -> None:
    if SAVECleanReasonRoute is None:
        pytest.fail("SAVECleanReasonRoute is not implemented")

    torch.manual_seed(31)
    action_bridge = nn.Linear(3, 2)
    route = SAVECleanReasonRoute(
        dim=8,
        reason_dim=3,
        action_dim=2,
        rank=2,
        reason_to_action=action_bridge,
    )
    foundation_reason = torch.randn(2, 3, requires_grad=True)
    reason_nodes = torch.randn(2, 3, 8, requires_grad=True)
    predicate_tokens = torch.randn(2, 4, 8, requires_grad=True)
    reliability = torch.rand(2, 3, requires_grad=True)
    output = route(
        reason_logits_calalign=foundation_reason,
        reason_nodes=reason_nodes,
        predicate_token=predicate_tokens,
        predicate_state_prob=torch.rand(2, 3, 4),
        factor_reliability=reliability,
        action_evidence_overlap=torch.rand(2, 3),
    )

    assert output["reason_logits_clean"].shape == (2, 3)
    assert output["action_logits_clean"].shape == (2, 2)
    assert torch.equal(output["reason_logits_clean"].detach(), foundation_reason.detach())

    loss = output["reason_logits_clean"].square().mean() + output["action_logits_clean"].square().mean()
    loss.backward()

    assert foundation_reason.grad is not None
    assert torch.count_nonzero(foundation_reason.grad) > 0
    assert reason_nodes.grad is not None
    assert torch.count_nonzero(reason_nodes.grad) > 0
    assert action_bridge.weight.grad is not None
    assert torch.count_nonzero(action_bridge.weight.grad) > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in route.parameters()
    )
    assert not output["reason_reliability"].requires_grad
    assert reliability.grad is None or torch.count_nonzero(reliability.grad) == 0


def test_combined_reason_decoder_preserves_live_clean_branch() -> None:
    if SAVEReasonDecoder is None:
        pytest.fail("SAVEReasonDecoder is not implemented")

    torch.manual_seed(32)
    model = SAVEReasonDecoder(dim=8, reason_dim=3, action_dim=2, rank=2, num_heads=2)
    foundation_reason = torch.randn(1, 3, requires_grad=True)
    reason_nodes = torch.randn(1, 3, 8, requires_grad=True)
    output = model(
        reason_logits_calalign=foundation_reason,
        reason_nodes=reason_nodes,
        global_field=torch.randn(1, 11, 8),
        detail_field=torch.randn(1, 11, 8),
        factor_measurement_token=torch.randn(1, 3, 8),
        factor_evidence_map=torch.rand(1, 3, 11),
        factor_reliability=torch.full((1, 3), 0.5),
        progress=1.0,
    )
    output["reason_logits_clean"].square().mean().backward()

    assert foundation_reason.grad is not None
    assert torch.count_nonzero(foundation_reason.grad) > 0
    assert reason_nodes.grad is not None
    assert torch.count_nonzero(reason_nodes.grad) > 0


def test_clean_reason_delta_cap_is_exactly_fifteen_percent_per_label_rms() -> None:
    route = SAVECleanReasonRoute(dim=8, reason_dim=3, action_dim=2, rank=2, num_heads=2)
    base = torch.tensor([[0.10, 1.0, 4.0], [-0.10, -1.0, -4.0]])
    with torch.no_grad():
        route.clean_gate_raw.fill_(10.0)
        route.semantic_head.weight.fill_(100.0)
        route.semantic_head.bias.fill_(100.0)

    output = route(
        reason_logits_calalign=base,
        reason_nodes=torch.randn(2, 3, 8),
        factor_reliability=torch.ones(2, 3),
    )
    expected_cap = 0.15 * base.square().mean(0).sqrt()

    torch.testing.assert_close(output["clean_logit_cap"], expected_cap)
    assert torch.all(output["reason_logits_clean_delta"].abs() <= expected_cap + 1e-7)
