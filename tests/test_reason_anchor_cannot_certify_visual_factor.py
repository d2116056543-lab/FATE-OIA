import inspect

from fate_oia.models.mosaic_continuous_credibility import visual_credibility_from_measurements


def test_visual_credibility_has_no_reason_label_input():
    names = set(inspect.signature(visual_credibility_from_measurements).parameters)
    assert "reason_labels" not in names
    assert "observed_reason" not in names

