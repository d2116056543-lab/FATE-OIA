from pathlib import Path


def test_transport_values_are_visual_only_by_source_contract():
    source = (Path(__file__).parents[1] / "fate_oia/models/vetra_visual_factor_transport.py").read_text()
    assert "named_values = visual_values" in source
    assert "unnamed_values = visual_values" in source
    assert "reason_nodes[:, reason_ids] + predicate_tokens" in source
    assert "named_values = reason" not in source
    assert "named_values = predicate" not in source
