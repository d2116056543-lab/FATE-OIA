import torch

from fate_oia.models.aie_predicate_naming import AIEPredicateNaming


def test_name_requires_confidence_margin_and_presence():
    module = AIEPredicateNaming(dim=32, num_predicates=32)
    out = module(torch.randn(1, 4, 4, 32), torch.full((1, 4, 4, 20), 0.05), torch.full((1, 32, 20), 0.05), torch.zeros(1, 32))
    assert bool((out["name_id"] == -1).all())

