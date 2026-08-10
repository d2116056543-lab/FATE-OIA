from vetra_test_utils import inputs, transport


def test_action_correction_respects_hard_cap():
    model = transport(); out = model(**inputs(), alpha=1.0)
    assert float(out["vetra_action_delta"].abs().max()) <= model.correction_cap + 1e-7
