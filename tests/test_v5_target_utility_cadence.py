from __future__ import annotations

import inspect

from fate_oia.engine.mosaic_icdor_adaptive_schedule import ICDORAdaptiveSchedule
from fate_oia.engine import train_acpr_mosaic_trust_icdor as trainer


def test_target_utility_has_epoch_zero_online_and_two_epoch_full_cadence() -> None:
    schedule = ICDORAdaptiveSchedule(pilot=False)
    assert schedule.online_target_probe_due(epoch=0)
    assert schedule.online_target_probe_due(epoch=1)
    assert schedule.full_target_audit_due(epoch=0)
    assert not schedule.full_target_audit_due(epoch=1)
    assert schedule.full_target_audit_due(epoch=2)


def test_trainer_uses_separate_online_and_full_target_audits() -> None:
    source = inspect.getsource(trainer)
    assert "online_target_probe_max_samples" in source
    assert "full_target_audit_due" in source
    assert "online_target_transfer" in source
    assert "full_target_transfer" in source
