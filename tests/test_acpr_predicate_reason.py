import torch

from fate_oia.models.acpr_predicate_reason import ACPRPredicateReasoner


def test_acpr_predicate_reason_delta_bounded():
    m = ACPRPredicateReasoner()
    out = m(torch.randn(2, 21, 384), torch.rand(2, 32), torch.randn(2, 32, 384))
    assert out["predicate_reason_delta"].shape == (2, 21)
    assert float(out["predicate_reason_delta"].abs().max()) <= 0.20
