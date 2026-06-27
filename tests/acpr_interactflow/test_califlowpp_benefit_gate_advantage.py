from __future__ import annotations

import torch

from fate_oia.acpr_interactflow.decision_ledger import DecisionLedgerHead


def test_benefit_gate_has_detached_advantage_target_and_gradients() -> None:
    ledger = DecisionLedgerHead(dim=16, num_actions=3)
    visual = torch.randn(2, 16)
    motion = torch.randn(2, 16)
    predicate = torch.randn(2, 16)
    factors = torch.randn(2, 5, 16)
    flow = torch.randn(2, 5, 3)
    soft = torch.softmax(torch.randn(2, 3), dim=-1)

    out = ledger(visual, motion, predicate, factors, flow, action_soft_target=soft)
    assert out.benefit_target is not None
    assert out.benefit_target.requires_grad is False

    assert out.benefit_target.shape == (2, 5, 1)
    loss = torch.nn.functional.binary_cross_entropy(out.benefit_gate.mean(-1, keepdim=True).clamp(1e-6, 1 - 1e-6), out.benefit_target)
    loss.backward()
    assert ledger.benefit_gate.weight.grad is not None
    assert torch.isfinite(ledger.benefit_gate.weight.grad).all()
