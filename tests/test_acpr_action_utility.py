import torch

from fate_oia.models.acpr_action_utility import ACPRActionUtility


def test_action_utility_zero_gates_exact_fallback():
    m = ACPRActionUtility(action_dim=4)
    fallback = torch.randn(3, 4)
    visual = torch.randn(3, 4)
    reason = torch.randn(3, 4)
    pred = torch.randn(3, 4)
    out = m(fallback, visual, reason, pred)
    assert torch.allclose(out["action_logits_utility"], fallback, atol=0.0, rtol=0.0)


def test_action_utility_gates_affect_only_selected_action_and_clamp():
    m = ACPRActionUtility(action_dim=4, max_r2a_delta=0.2, max_pred_delta=0.05)
    fallback = torch.zeros(2, 4)
    visual = torch.zeros(2, 4)
    reason = torch.full((2, 4), 10.0)
    pred = torch.full((2, 4), -10.0)
    m.set_gates(r2a_gate=torch.tensor([0, 1, 0, 0.0]), pred_gate=torch.tensor([0, 0, 0, 1.0]))
    out = m(fallback, visual, reason, pred)
    assert torch.allclose(out["action_logits_utility"][:, 0], fallback[:, 0])
    assert torch.allclose(out["action_logits_utility"][:, 2], fallback[:, 2])
    assert torch.allclose(out["action_logits_utility"][:, 1], torch.full((2,), 0.2))
    assert torch.allclose(out["action_logits_utility"][:, 3], torch.full((2,), -0.05))
    assert out["action_r2a_delta"].abs().max() <= 0.20001
    assert out["action_predicate_delta"].abs().max() <= 0.05001
