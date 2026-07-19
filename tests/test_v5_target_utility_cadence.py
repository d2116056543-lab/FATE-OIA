from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from fate_oia.engine.mosaic_icdor_adaptive_schedule import ICDORAdaptiveSchedule
from fate_oia.engine import train_acpr_mosaic_trust_icdor as trainer


def test_target_utility_has_epoch_zero_online_and_two_epoch_full_cadence() -> None:
    schedule = ICDORAdaptiveSchedule(pilot=False)
    assert schedule.online_target_probe_due(epoch=0)
    assert schedule.online_target_probe_due(epoch=1)
    assert schedule.full_target_audit_due(epoch=0)
    assert not schedule.full_target_audit_due(epoch=1)
    assert schedule.full_target_audit_due(epoch=2)
    assert trainer._should_collect_full_target_transfer(pilot=True, full_target_audit_due=True)
    assert not trainer._should_collect_full_target_transfer(pilot=True, full_target_audit_due=False)


def test_trainer_uses_separate_online_and_full_target_audits() -> None:
    source = inspect.getsource(trainer)
    assert "online_target_probe_max_samples" in source
    assert "full_target_audit_due" in source
    assert "online_target_transfer" in source
    assert "full_target_transfer" in source


def test_pilot_online_target_probe_is_bounded_to_v5_lightweight_audit() -> None:
    config_path = Path(__file__).parents[1] / "configs" / (
        "fate_oia_train_360x640_acpr_mosaic_trust_v5_credo_map_pilot.yaml"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    # V5 reserves the 512-row bootstrap audit for the every-two-epoch full pass.
    assert 28 <= config["runtime"]["online_target_probe_max_samples"] <= 56
