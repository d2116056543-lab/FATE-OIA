from vetra_test_utils import inputs, transport


def test_counter_named_factors_only_use_contradictory_grammar_edges():
    model, data = transport(), inputs(batch=1)
    values, _, _ = model._visual_values(data["patch_tokens_by_layer_raw"], data["predicate_attention"])
    _, _, _, _, meta = model._factor_bank(1, values, data["reason_nodes_primary"], data["predicate_tokens"])
    reasons = meta["reason_ids"]
    assert bool(model.grammar_contradictory_mask[reasons, meta["predicate_ids"][:len(reasons)]].all())
