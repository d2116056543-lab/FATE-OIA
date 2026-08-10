from vetra_test_utils import inputs, transport


def test_every_predicate_layer_has_unnamed_visual_fallback():
    model, data = transport(), inputs(batch=1)
    out = model(**data)
    assert out["support_meta"]["unnamed_count"] == len(model.grammar_positive_mask[0]) * 3
    assert out["counter_meta"]["unnamed_count"] == len(model.grammar_positive_mask[0]) * 3
