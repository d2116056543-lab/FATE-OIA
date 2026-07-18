from pathlib import Path

from fate_oia.engine.mosaic_icdor_adaptive_schedule import ICDORAdaptiveSchedule


def test_hidden_pu_failure_only_disables_pu():
    schedule = ICDORAdaptiveSchedule(pilot=True)
    schedule.set_pu_enabled(False, reason="hidden_margin_negative")
    phase = schedule.phase()
    assert phase.route_mode == "shadow"
    assert phase.latent_enabled is True
    assert phase.pu_enabled is False


def test_trainer_wires_hidden_margin_only_to_the_mutable_pu_gate():
    source = Path("fate_oia/engine/train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert 'adaptive_schedule.set_pu_enabled(False, reason="hidden_recovery_margin_nonpositive")' in source
    assert "action shadow route remain active" in source
