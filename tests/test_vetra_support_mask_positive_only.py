from vetra_test_utils import inputs, transport


def test_support_named_factors_only_use_positive_grammar_edges():
    model, data = transport(), inputs(batch=1)
    values, _, _ = model._visual_values(data["patch_tokens_by_layer_raw"], data["predicate_attention"])
    _, _, allowed, _, meta = model._factor_bank(0, values, data["reason_nodes_primary"], data["predicate_tokens"])
    reasons = meta["reason_ids"]
    assert bool(model.grammar_positive_mask[reasons, meta["predicate_ids"][:len(reasons)]].all())
    assert allowed.shape[0] == 4
