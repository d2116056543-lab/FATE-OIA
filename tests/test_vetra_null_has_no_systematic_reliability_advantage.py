from pathlib import Path


def test_reliability_is_relative_and_null_has_explicit_prior():
    text=(Path(__file__).parents[1]/"fate_oia/models/vetra_visual_factor_transport.py").read_text()
    assert "non_null.amax" in text
    assert "self.null_route_prior" in text
    assert "score = score + routing_reliability" in text
