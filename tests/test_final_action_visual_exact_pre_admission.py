from __future__ import annotations

from fate_oia.engine.mosaic_icdor_adaptive_schedule import ICDORAdaptiveSchedule


def test_final_action_visual_exact_pre_admission() -> None:
    schedule = ICDORAdaptiveSchedule(pilot=False)
    assert schedule.policy().route_is_final is False
    assert schedule.policy().route_mode == "shadow"
