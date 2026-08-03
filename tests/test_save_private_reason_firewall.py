import pytest
import torch
from torch import nn

try:
    from fate_oia.models.save_reason_decoder import (
        SAVECleanReasonRoute,
        SAVEPrivateReasonDecoder,
    )
except ImportError:
    SAVECleanReasonRoute = None
    SAVEPrivateReasonDecoder = None


def test_benchmark_reason_loss_cannot_reach_clean_action_or_foundation() -> None:
    if SAVECleanReasonRoute is None or SAVEPrivateReasonDecoder is None:
        pytest.fail("SAVE reason routes are not implemented")

    torch.manual_seed(41)
    action_bridge = nn.Linear(3, 2)
    clean = SAVECleanReasonRoute(
        dim=8,
        reason_dim=3,
        action_dim=2,
        rank=2,
        reason_to_action=action_bridge,
    )
    private = SAVEPrivateReasonDecoder(dim=8, action_dim=2, reason_dim=3, num_heads=2)
    foundation_reason = torch.randn(1, 3, requires_grad=True)
    reason_nodes = torch.randn(1, 3, 8, requires_grad=True)
    global_field = torch.randn(1, 13, 8, requires_grad=True)
    detail_field = torch.randn(1, 13, 8, requires_grad=True)
    factor_tokens = torch.randn(1, 3, 8, requires_grad=True)
    factor_map = torch.rand(1, 3, 13, requires_grad=True)
    clean_output = clean(
        reason_logits_calalign=foundation_reason,
        reason_nodes=reason_nodes,
        factor_reliability=torch.full((1, 3), 0.5),
    )
    private_output = private(
        reason_logits_clean=clean_output["reason_logits_clean"],
        global_field=global_field,
        detail_field=detail_field,
        factor_measurement_token=factor_tokens,
        factor_evidence_map=factor_map,
        factor_reliability=torch.full((1, 3), 0.5),
        progress=1.0,
    )
    private_output["reason_logits_benchmark"].square().mean().backward()

    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in private.parameters()
    )
    for parameter in clean.parameters():
        assert parameter.grad is None
    assert action_bridge.weight.grad is None
    assert foundation_reason.grad is None
    assert reason_nodes.grad is None
    assert global_field.grad is None
    assert detail_field.grad is None
    assert factor_tokens.grad is None
    assert factor_map.grad is None
