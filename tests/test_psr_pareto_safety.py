from __future__ import annotations

import torch

from fate_oia.models.psr_oia_router import ParetoSafetySelector, evidence_reliability


def test_pareto_selector_falls_back_when_action_worse_and_evidence_zero():
    labels_a = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=torch.float32)
    labels_r = torch.randint(0, 2, (3, 21)).float()
    base = labels_a * 8 - 4
    worse = -base
    reason = torch.randn(3, 21)
    selected, meta = ParetoSafetySelector().guard_action(worse, base, reason, labels_a, labels_r)
    assert meta["pareto_action_fallback"] is True
    assert torch.allclose(selected, base)
    assert evidence_reliability(0.2, 0.2) == 0.0
    assert evidence_reliability(0.1, 0.3) == 0.0
    assert evidence_reliability(0.5, 0.3) > 0.0
