from __future__ import annotations

import importlib

import pytest


def _foundation_ready() -> dict[str, object]:
    return {
        "factor_audit_complete": True,
        "factor_audit_exception": False,
        "unknown_abstained": True,
        "certified_route_group_count_per_action": [1, 1, 1, 1],
        "reachable_reason_count": 15,
        "certificate_tier_jaccard": 0.95,
        "saturation_fraction": 0.10,
        "diagnostics_finite": True,
    }


def test_adaptive_schedule_enforces_state_side_effects_and_two_epoch_readiness() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_adaptive_schedule")
    machine = module.ICDORAdaptiveSchedule(pilot=True)
    assert machine.state == "FOUNDATION"
    policy = machine.policy()
    assert policy.route_mode == "off"
    assert policy.action_rank_weight == 0.0 and policy.reason_rank_weight == 0.0
    assert policy.write_provisional_certificate is True
    for epoch in range(3):
        machine.update(epoch=epoch, train_audit_metrics=_foundation_ready(), train_calib_metrics={})
    assert machine.state == "DUAL_REASON_SHADOW"
    assert machine.policy().freeze_factor_and_prototypes is True
    assert machine.policy().route_mode == "shadow"
    with pytest.raises(ValueError, match="test"):
        machine.update(epoch=4, train_audit_metrics=_foundation_ready(), train_calib_metrics={}, test_metrics={})


def test_adaptive_schedule_fails_closed_at_state_maximum() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_adaptive_schedule")
    machine = module.ICDORAdaptiveSchedule(pilot=False)
    bad = _foundation_ready()
    bad["factor_audit_complete"] = False
    for epoch in range(6):
        machine.update(epoch=epoch, train_audit_metrics=bad, train_calib_metrics={})
    assert machine.failed_closed is True
    assert machine.full_train_eligible is False
