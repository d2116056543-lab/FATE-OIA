from __future__ import annotations

import importlib

import pytest


def _foundation_ready() -> dict[str, object]:
    return {
        "continuous_credibility_available": True,
        "observable_cV_gt_030": 6,
        "factor_audit_complete": True,
        "factor_audit_exception": False,
        "unknown_abstained": True,
        "certified_route_group_count_per_action": [1, 1, 1, 1],
        "reachable_reason_count": 15,
        "certificate_tier_jaccard": 0.95,
        "saturation_fraction": 0.10,
        "diagnostics_finite": True,
    }


def _core_ready() -> dict[str, object]:
    return {"source_split": "train_core", "finite": True}


def _calib_ready() -> dict[str, object]:
    return {"source_split": "train_calib", "finite": True}


def test_adaptive_schedule_enforces_state_side_effects_and_two_epoch_readiness() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_adaptive_schedule")
    machine = module.ICDORAdaptiveSchedule(pilot=True)
    assert machine.state == "FOUNDATION"
    policy = machine.policy()
    assert policy.route_mode == "shadow"
    assert policy.action_rank_weight == 0.10 and policy.reason_rank_weight == 0.05
    assert policy.write_provisional_certificate is True
    for epoch in range(3):
        machine.update(
            epoch=epoch,
            train_audit_metrics=_foundation_ready(),
            train_calib_metrics=_calib_ready(),
            train_core_metrics=_core_ready(),
        )
    assert machine.state == "DUAL_REASON_SHADOW"
    assert machine.policy().freeze_factor_and_prototypes is True
    assert machine.policy().route_mode == "shadow"
    with pytest.raises(ValueError, match="test"):
        machine.update(
            epoch=4,
            train_audit_metrics=_foundation_ready(),
            train_calib_metrics=_calib_ready(),
            train_core_metrics=_core_ready(),
            test_metrics={},
        )


def test_adaptive_schedule_keeps_shadow_learning_when_certificate_is_missing() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_adaptive_schedule")
    machine = module.ICDORAdaptiveSchedule(pilot=False)
    bad = _foundation_ready()
    bad["factor_audit_complete"] = False
    for epoch in range(6):
        machine.update(
            epoch=epoch,
            train_audit_metrics=bad,
            train_calib_metrics=_calib_ready(),
            train_core_metrics=_core_ready(),
        )
    assert machine.failed_closed is False
    assert machine.full_train_eligible is False
    assert machine.policy().route_mode == "shadow"


def test_adaptive_schedule_requires_finite_train_calib_for_transition() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_adaptive_schedule")
    machine = module.ICDORAdaptiveSchedule(pilot=True)
    bad_calib = {"source_split": "train_calib", "finite": False}
    for epoch in range(6):
        machine.update(
            epoch=epoch,
            train_audit_metrics=_foundation_ready(),
            train_calib_metrics=bad_calib,
            train_core_metrics=_core_ready(),
        )
    assert machine.state == "FOUNDATION"
    assert machine.failed_closed is False
    assert machine.history[-1]["ready"] is False
    assert machine.policy().route_mode == "shadow"
