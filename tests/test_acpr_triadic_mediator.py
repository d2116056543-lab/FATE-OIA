import torch

from fate_oia.models.acpr_triadic_mediator import ACPRTriadicMediator


def test_triadic_zero_init_equivalence_and_bounds():
    m = ACPRTriadicMediator(action_dim=4, reason_dim=21, num_predicates=32, max_action_delta=0.10)
    av = torch.randn(2, 4)
    ar = torch.randn(2, 4)
    rl = torch.randn(2, 21)
    pp = torch.sigmoid(torch.randn(2, 32))
    out = m(av, ar, rl, pp)
    assert torch.max(torch.abs(out["triadic_action_delta"])) < 1e-6
    assert torch.allclose(out["action_reason_logits_triadic"], ar, atol=1e-6)
    assert out["triadic_reason_support"].shape == (2, 4, 21)
    assert out["triadic_predicate_support"].shape == (2, 4, 32)


def test_triadic_masks_come_from_grammar_not_modulo_or_all_one():
    m = ACPRTriadicMediator(action_dim=4, reason_dim=21, num_predicates=32, max_action_delta=0.10)
    assert float(m.reason_predicate_positive_mask[0, 1]) == 1.0
    assert float(m.reason_predicate_positive_mask[0, 21]) == 1.0
    active_r0 = set(torch.where(m.reason_predicate_positive_mask[0] > 0)[0].tolist())
    assert active_r0 != {0, 3}
    assert not torch.all(m.action_predicate_support_mask == 1)


def test_triadic_depends_on_reason_and_predicate_after_scale_change():
    m = ACPRTriadicMediator(action_dim=4, reason_dim=21, num_predicates=32, max_action_delta=0.10)
    with torch.no_grad():
        m.delta_scale.fill_(1.0)
    ar = torch.zeros(1, 4)
    av = torch.zeros(1, 4)
    pp1 = torch.zeros(1, 32); pp1[:, 1] = 1
    pp2 = torch.zeros(1, 32); pp2[:, 21] = 1
    r1 = torch.zeros(1, 21); r1[:, 0] = 5
    r2 = torch.zeros(1, 21); r2[:, 13] = 5
    o1 = m(av, ar, r1, pp1)["triadic_action_delta"]
    o2 = m(av, ar, r2, pp2)["triadic_action_delta"]
    assert torch.max(torch.abs(o1)) <= 0.100001
    assert not torch.allclose(o1, o2)
