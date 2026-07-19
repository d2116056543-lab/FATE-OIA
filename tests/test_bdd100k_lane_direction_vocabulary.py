from __future__ import annotations

from fate_oia.datasets.mosaic_icdor_grounding import is_supported_lane_direction


def test_bdd100k_lane_direction_vocabulary() -> None:
    assert is_supported_lane_direction("parallel")
    assert is_supported_lane_direction("vertical")
    assert not is_supported_lane_direction("left")
    assert not is_supported_lane_direction("right")
