import torch

from fate_oia.models.save_utility_bridge import SAVEUtilityBridge, select_teacher_predicates


def test_teacher_predicate_candidates_ignore_current_utility_predictions() -> None:
    candidate_weight = torch.tensor(
        [[[0.05, 0.60, 0.20, 0.15], [0.10, 0.10, 0.70, 0.10]]]
    )
    reliability = torch.tensor([[0.5, 1.0, 0.8]])
    overlap = torch.tensor([[[0.1, 0.9, 0.2], [0.3, 0.2, 0.8]]])
    utility_a = torch.tensor([[[100.0, -100.0, 20.0], [-50.0, 40.0, -30.0]]])
    utility_b = -utility_a

    selected_a = select_teacher_predicates(
        candidate_weight,
        reliability,
        overlap,
        utility_logit=utility_a,
        max_predicates=2,
    )
    selected_b = select_teacher_predicates(
        candidate_weight,
        reliability,
        overlap,
        utility_logit=utility_b,
        max_predicates=2,
    )

    assert torch.equal(selected_a, selected_b)
    assert selected_a.tolist() == [[[1, 2], [2, 1]]]


def test_candidate_and_utility_equations_are_locked_and_independent() -> None:
    torch.manual_seed(31)
    bridge = SAVEUtilityBridge(dim=8, action_dim=2, factor_dim=3)
    with torch.no_grad():
        bridge.candidate_query.weight.zero_()
        bridge.candidate_key.weight.zero_()
        bridge.null_candidate_key.zero_()
        bridge.candidate_bias.zero_()

    action = torch.randn(2, 2, 8)
    predicate = torch.randn(2, 3, 8)
    state = torch.randn(2, 3, 8)
    reliability = torch.rand(2, 3)
    overlap = torch.rand(2, 2, 3)
    similarity = torch.rand(2, 2, 3)
    first = bridge(action, predicate, state, None, reliability, None, overlap, similarity)
    first_candidate = first["predicate_candidate_weight"].detach().clone()

    with torch.no_grad():
        for parameter in bridge.state_projection.parameters():
            parameter.add_(0.7)
        for parameter in bridge.utility_action_projection.parameters():
            parameter.add_(0.3)
        for parameter in bridge.utility_predicate_projection.parameters():
            parameter.sub_(0.2)
        bridge.utility_mlp[-1].bias.add_(4.0)
    second = bridge(action, predicate, state, None, reliability, None, overlap, similarity)

    torch.testing.assert_close(first_candidate, second["predicate_candidate_weight"])
    torch.testing.assert_close(
        second["predicate_candidate_weight"].sum(-1),
        torch.ones(2, 2),
        atol=1e-6,
        rtol=0,
    )
    assert second["predicate_candidate_weight"].shape[-1] == 4
    assert torch.all(second["predicate_candidate_weight_null"] > 0)
    assert not torch.allclose(first["utility_logit"], second["utility_logit"])
    assert bridge.rank == 32
    assert bridge.utility_mlp[0].in_features == 32 + 4

    projected_state = bridge.state_projection(state)
    predicate_rank = bridge.utility_predicate_projection(predicate + projected_state)
    action_rank = bridge.utility_action_projection(action)
    bilinear = action_rank.unsqueeze(2) * predicate_rank.unsqueeze(1)
    scalars = torch.stack(
        (
            reliability.unsqueeze(1).expand(-1, 2, -1),
            overlap,
            similarity,
            second["predicate_candidate_weight_real"],
        ),
        dim=-1,
    )
    expected = bridge.utility_mlp(torch.cat((bilinear, scalars), dim=-1)).squeeze(-1)
    torch.testing.assert_close(second["utility_logit"], expected)
    required = {
        "predicate_candidate_weight",
        "predicate_candidate_weight_real",
        "predicate_candidate_weight_null",
        "utility_logit",
        "utility_prob",
        "utility_logit_with_null",
        "utility_prob_with_null",
        "utility_rank",
        "utility_teacher_due",
        "teacher_plan",
        "utility_teacher_target",
        "utility_counterfactual_weight",
        "utility_dense_auxiliary_weight",
    }
    assert required.issubset(second)
    torch.testing.assert_close(
        second["utility_counterfactual_weight"], torch.tensor(0.10)
    )
    torch.testing.assert_close(
        second["utility_dense_auxiliary_weight"], torch.tensor(0.02)
    )
