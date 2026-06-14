from fate_oia.models.acpr_reason_grammar import ACPRReasonGrammar


def test_acpr_reason_grammar_no_placeholders():
    g = ACPRReasonGrammar("configs/acpr_reason_predicate_grammar.yaml")
    assert len(g.action_names) == 4
    assert len(g.reason_names) == 21
    assert all(not name.startswith("reason_") for name in g.reason_names)
    assert set([12, 9, 5, 14, 6, 11, 10, 13]).issubset(set(g.tail_indices))
