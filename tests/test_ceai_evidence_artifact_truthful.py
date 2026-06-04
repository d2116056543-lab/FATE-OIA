from fate_oia.utils.ceai_artifacts import make_selected_vs_random_evidence_stats


def test_uncomputed_evidence_artifact_is_explicitly_unavailable():
    stats = make_selected_vs_random_evidence_stats(None, None, computed=False)
    assert stats["available"] is False
    assert stats["reason"] == "not_computed_in_ceai_v1_1"
    assert stats["selected_mean"] is None
    assert stats["random_mean"] is None
    assert stats["evidence_gate_active"] is False


def test_fake_zero_zero_artifact_is_not_allowed_when_available():
    stats = make_selected_vs_random_evidence_stats(0.0, 0.0, computed=True)
    assert stats["available"] is False
    assert stats["reason"] == "degenerate_zero_zero_not_evidence"
