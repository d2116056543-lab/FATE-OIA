from fate_oia.models.aie_evidence_interface import AIEEvidenceInterface
from fate_oia.models.aie_predicate_naming import AIEPredicateNaming


def test_naming_table_contains_only_real_predicates():
    evidence = AIEEvidenceInterface(dim=32, num_predicates=32)
    naming = AIEPredicateNaming(dim=32, num_predicates=32)
    assert evidence.predicate_keys.shape == (32, 64)
    assert not hasattr(naming, "predicate_keys")
