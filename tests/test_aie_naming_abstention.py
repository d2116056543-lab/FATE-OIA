import torch

from fate_oia.models.aie_predicate_naming import AIEPredicateNaming


def test_naming_abstains_without_quality_and_has_no_null_class():
    module = AIEPredicateNaming(dim=32, num_predicates=32)
    maps = torch.softmax(torch.randn(2, 4, 4, 20), -1)
    pattn = torch.softmax(torch.randn(2, 32, 20), -1)
    pprob = torch.zeros(2, 32)
    compatibility = torch.rand(2, 4, 4, 32)
    out = module(maps, pattn, pprob, compatibility)
    assert not hasattr(module, "predicate_keys")
    assert torch.equal(out["name_id"], torch.full((2, 4, 4), -1, dtype=torch.long))

