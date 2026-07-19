from pathlib import Path

from fate_oia.engine.mosaic_icdor_adaptive_schedule import ICDORAdaptiveSchedule


def test_v5_hidden_pu_failure_is_diagnostic_only_and_keeps_labelwise_latent_learning():
    schedule = ICDORAdaptiveSchedule(pilot=True)
    schedule.set_pu_enabled(False, reason="hidden_margin_negative")
    phase = schedule.phase()
    assert phase.route_mode == "shadow"
    assert phase.latent_enabled is True
    assert phase.pu_enabled is True
    assert schedule.pu_disable_reason == "hidden_margin_negative"


def test_trainer_keeps_the_global_hidden_margin_out_of_the_latent_core_gate():
    source = Path("fate_oia/engine/train_acpr_mosaic_trust_icdor.py").read_text(encoding="utf-8")
    assert "pu_gate=model.reason_pu_gate" in source
    assert "latent_reason_core_loss" in source
    # The call persists the failed global diagnostic for artifacts. The V5
    # schedule deliberately ignores it for latent-core trainability.
    assert 'adaptive_schedule.set_pu_enabled(False, reason="hidden_recovery_margin_nonpositive")' in source
