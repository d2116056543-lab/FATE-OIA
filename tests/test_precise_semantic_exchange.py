from pathlib import Path

import torch

from fate_oia.models.precise_semantic_exchange import PRECISESemanticExchange
from fate_oia.utils.precise_schema import load_evidence_fields, load_reason_semantics


ROOT = Path(__file__).resolve().parents[1]


def _model():
    return PRECISESemanticExchange(load_evidence_fields(ROOT / "configs" / "precise_evidence_fields.yaml"), load_reason_semantics(ROOT / "configs" / "precise_reason_semantics.yaml"))


def test_zero_explicit_reliability_forces_zero_certified_messages():
    model = _model()
    output = model(torch.randn(2, 4, 384), torch.randn(2, 21, 384), torch.randn(2, 10, 384), torch.zeros(2, 10))
    assert output["action_exchange_delta"].abs().max().item() < 1e-7
    assert output["reason_exchange_delta"].abs().max().item() < 1e-7


def test_action_exchange_blocks_reason_owner_gradient():
    model = _model()
    action = torch.randn(1, 4, 384, requires_grad=True)
    reason = torch.randn(1, 21, 384, requires_grad=True)
    output = model(action, reason, torch.randn(1, 10, 384), torch.full((1, 10), 0.7))
    output["action_exchange_delta"].square().mean().backward()
    assert reason.grad is None
    assert action.grad is not None


def test_wrong_target_ratio_uses_action_reason_groups_not_min_max_weights():
    model = _model()
    assert model.action_reason_compatibility.shape == (4, 21)
    assert model.action_reason_compatibility.sum(dim=1).tolist() == [3, 6, 6, 6]
    output = model(torch.randn(2, 4, 384), torch.randn(2, 21, 384), torch.randn(2, 10, 384), torch.full((2, 10), 0.7))
    expected = output["wrong_target_message_mass"] / output["correct_target_message_mass"].clamp_min(1e-8)
    assert torch.allclose(output["wrong_target_message_ratio"], expected)
