from fate_oia.models.mosaic_continuous_credibility import absence_polarity


def test_no_lane_uses_absence_polarity():
    assert absence_polarity(presence=0.0, observability=1.0) == 1.0
    assert absence_polarity(presence=1.0, observability=1.0) == 0.0
    assert absence_polarity(presence=0.0, observability=0.0) == 0.0

