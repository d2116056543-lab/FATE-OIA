from __future__ import annotations

from fate_oia.datasets.mosaic_icdor_grounding import reliable_absence_observation


def test_reliable_absence_observability() -> None:
    observation = reliable_absence_observation(source_complete=True, region_visible=True, no_footpoint=True)
    assert observation == {"presence": 0.0, "observability": 1.0, "known": True}
