import torch
from fate_oia.models.acpr_predicate_conditioned_threshold import ACPRPredicateConditionedThreshold


def test_threshold_delta_zero_init_matches_base():
    head = ACPRPredicateConditionedThreshold(action_dim=4, reason_dim=21, num_predicates=32)
    action = torch.randn(3, 4)
    reason = torch.randn(3, 21)
    pred = torch.randn(3, 32)
    old = head(action, reason, predicate_context=None)
    new = head(action, reason, predicate_context=pred)
    assert torch.allclose(old["logits_deploy"], new["logits_deploy"], atol=1e-6)
    assert torch.max(torch.abs(new["threshold_delta"])) < 1e-6


def test_threshold_delta_bounded_after_perturb():
    head = ACPRPredicateConditionedThreshold(action_dim=4, reason_dim=21, num_predicates=32, threshold_delta_max=0.10)
    with torch.no_grad():
        head.predicate_to_delta[-1].weight.fill_(1.0)
    out = head(torch.zeros(2, 4), torch.zeros(2, 21), torch.ones(2, 32))
    assert torch.max(torch.abs(out["threshold_delta"])) <= 0.100001
