import torch

from fate_oia.models.acpr_action_candidates import ACPRActionCandidates


def test_candidate_dict_and_zero_gate_fallback_invariant():
    module = ACPRActionCandidates(action_dim=4, max_pred_delta=0.05)
    fallback = torch.randn(3, 4)
    visual = torch.randn(3, 4)
    reason = torch.randn(3, 4)
    theta = torch.tensor([0.1, -0.2, 0.3, -0.4])
    pred_delta = torch.full((3, 4), 0.20)

    out = module(fallback, visual, reason, theta, pred_delta, probe_mode=True)

    assert set(["fallback", "visual", "reason", "blend", "predicate", "blend_predicate", "utility_final"]).issubset(out)
    assert torch.allclose(out["utility_final"], fallback, atol=1e-6)
    assert torch.all(out["predicate_delta_clipped"].abs() <= 0.05001)
    assert torch.all((out["blend_gamma"] >= 0) & (out["blend_gamma"] <= 1))
    assert torch.allclose(out["visual"], visual - theta.view(1, 4), atol=1e-6)
    assert torch.allclose(out["reason"], reason - theta.view(1, 4), atol=1e-6)


def test_selected_candidate_changes_only_selected_action():
    module = ACPRActionCandidates(action_dim=4)
    fallback = torch.zeros(2, 4)
    visual = torch.ones(2, 4)
    reason = torch.full((2, 4), 2.0)
    theta = torch.zeros(4)
    module.set_selected_candidates(torch.tensor([-1, 0, -1, -1]), torch.tensor([0.0, 1.0, 0.0, 0.0]))

    out = module(fallback, visual, reason, theta)

    expected = fallback.clone()
    expected[:, 1] = 1.0
    assert torch.allclose(out["utility_final"], expected, atol=1e-6)

