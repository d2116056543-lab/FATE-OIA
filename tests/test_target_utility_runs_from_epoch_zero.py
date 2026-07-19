from __future__ import annotations

from fate_oia.engine.mosaic_icdor_adaptive_schedule import ICDORAdaptiveSchedule


def test_target_utility_runs_from_epoch_zero() -> None:
    schedule = ICDORAdaptiveSchedule(pilot=True)
    assert schedule.policy().enable_interventions is True
    assert schedule.state == "JOINT_SHADOW"
