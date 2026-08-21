import torch

from fate_oia.engine.evaluate_tida_oia import select_predicate_intervention_indices


def test_selected_predicates_follow_logit_contribution_not_attention_mass():
    route = torch.tensor([[[0.90, 0.39, 0.10, 0.40, 0.80]]])
    contribution = torch.tensor([[[0.01, 0.02, 0.03, 0.50, 0.04]]])

    selected, matched = select_predicate_intervention_indices(route, contribution, count=1)

    assert selected.tolist() == [3]
    assert matched.tolist() == [1]
