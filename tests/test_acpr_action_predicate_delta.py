import torch

from fate_oia.models.acpr_action_predicate_delta import ACPRActionPredicateDelta


def test_action_predicate_delta_tiny_nonzero_init_and_shape():
    m = ACPRActionPredicateDelta(dim=8, num_predicates=5, action_dim=4, hidden_dim=4)
    action_nodes = torch.randn(2, 4, 8)
    predicates = torch.rand(2, 5)
    out = m(action_nodes, predicates)
    assert out["predicate_action_delta"].shape == (2, 4)
    assert torch.isfinite(out["predicate_action_delta"]).all()
    assert out["predicate_action_delta"].abs().max() <= 0.05001
    assert out["predicate_action_delta"].abs().sum() > 0


def test_action_predicate_delta_detach_inputs_default():
    m = ACPRActionPredicateDelta(dim=8, num_predicates=5, action_dim=4, hidden_dim=4)
    action_nodes = torch.randn(2, 4, 8, requires_grad=True)
    predicates = torch.rand(2, 5, requires_grad=True)
    out = m(action_nodes, predicates)
    out["predicate_action_delta_raw"].sum().backward()
    assert action_nodes.grad is None
    assert predicates.grad is None
