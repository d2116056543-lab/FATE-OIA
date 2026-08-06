from fate_oia.models.aie_predicate_naming import AIEPredicateNaming


def test_naming_table_contains_only_real_predicates():
    module = AIEPredicateNaming(dim=32, num_predicates=32)
    assert module.predicate_keys.shape == (32, 64)

