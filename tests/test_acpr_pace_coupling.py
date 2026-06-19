import torch

from fate_oia.models.acpr_predicate_action_coupling import ACPRPredicateActionCoupling


def test_pace_coupling_zero_strength_is_legacy():
    torch.manual_seed(1)
    b = 3
    visual = torch.randn(b, 4)
    reason = torch.randn(b, 4)
    gate = torch.sigmoid(torch.randn(b, 4))
    delta = torch.randn(b, 21) * 0.1
    weight = torch.randn(4, 21)
    mod = ACPRPredicateActionCoupling(coupling_strength=0.0)
    out = mod(visual, reason, gate, delta, weight)
    legacy = gate * visual + (1 - gate) * reason
    assert torch.allclose(out["action_logits_pace"], legacy, atol=1e-6, rtol=1e-6)
    assert torch.allclose(out["predicate_action_delta_bounded"], torch.zeros_like(out["predicate_action_delta_bounded"]))


def test_pace_coupling_contributions_sum_exactly():
    torch.manual_seed(2)
    b = 5
    visual = torch.randn(b, 4)
    reason = torch.randn(b, 4)
    gate = torch.sigmoid(torch.randn(b, 4))
    delta = torch.randn(b, 21)
    weight = torch.randn(4, 21)
    mod = ACPRPredicateActionCoupling(coupling_strength=1.7, max_action_delta=0.2)
    out = mod(visual, reason, gate, delta, weight)
    assert out["predicate_action_delta_bounded"].abs().max() <= 0.200001
    raw_sum = out["predicate_reason_action_contrib_raw"].sum(-1)
    assert torch.allclose(raw_sum, out["predicate_action_delta_raw"], atol=1e-6, rtol=1e-6)
    final_sum = out["predicate_reason_action_contrib_final"].sum(-1)
    assert torch.allclose(final_sum, out["action_logits_pace"] - out["action_logits_legacy"], atol=1e-6, rtol=1e-5)
