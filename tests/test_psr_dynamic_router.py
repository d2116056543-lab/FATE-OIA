from __future__ import annotations

import torch

from fate_oia.models.psr_oia_router import DynamicMarginEntropyRouter


def test_dynamic_router_responds_to_margin_and_evidence():
    reason_a = torch.zeros(4, 21)
    reason_e = torch.ones(4, 21) * 3.0
    action_a = torch.ones(4, 4) * 5.0
    action_e = torch.randn(4, 4)
    out_no_ev = DynamicMarginEntropyRouter()(action_a, reason_a, action_e, reason_e, evidence_rel=0.0)
    assert out_no_ev.alpha_reason.max() == 1.0
    assert out_no_ev.alpha_action.max() == 0.0
    uncertain_action = torch.zeros(4, 4)
    out_ev = DynamicMarginEntropyRouter()(uncertain_action, reason_a, action_e, reason_e, evidence_rel=1.0)
    assert out_ev.alpha_action.max() > 0.0
