from __future__ import annotations

import importlib

import pytest


def _v5_ready() -> dict[str, object]:
    return {
        "continuous_credibility_available": True,
        "observable_cV_gt_030": 6,
        "direct_action_map": 0.70,
        "direct_reason_map": 0.35,
        "direct_action_map_stable_or_improving": True,
        "direct_reason_map_stable_or_improving": True,
        "mean_abs_cV_delta": 0.04,
        "nonzero_semantic_reason_count": 10,
        "final_action_visual_exact": True,
        "route_strength_ratio": 0.02,
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


def test_v5_schedule_starts_joint_shadow_and_promotes_only_after_eight_ready_epochs() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_adaptive_schedule")
    machine = module.ICDORAdaptiveSchedule(pilot=True)
    assert machine.state == "JOINT_SHADOW"
    assert machine.LIMITS["JOINT_SHADOW"] == (8, 12)
    assert machine.LIMITS["ADMISSION_CONSOLIDATION"] == (2, 2)
    policy = machine.policy()
    assert policy.route_mode == "shadow"
    assert policy.action_rank_weight == 0.10 and policy.reason_rank_weight == 0.05
    assert machine.online_target_probe_due(epoch=0) is True
    assert machine.full_target_audit_due(epoch=0) is True
    for epoch in range(7):
        machine.update(
            epoch=epoch,
            train_audit_metrics=_v5_ready(),
            train_calib_metrics=_calib_ready(),
            train_core_metrics=_core_ready(),
        )
    assert machine.state == "JOINT_SHADOW"
    machine.update(
        epoch=7,
        train_audit_metrics=_v5_ready(),
        train_calib_metrics=_calib_ready(),
        train_core_metrics=_core_ready(),
    )
    assert machine.state == "ADMISSION_CONSOLIDATION"
    assert machine.policy().freeze_certificate is True
    assert machine.policy().route_mode == "admitted"
    with pytest.raises(ValueError, match="test"):
        machine.update(
            epoch=4,
                train_audit_metrics=_v5_ready(),
            train_calib_metrics=_calib_ready(),
            train_core_metrics=_core_ready(),
            test_metrics={},
        )


def test_v5_schedule_does_not_block_learning_or_admission_on_a_global_certificate() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_adaptive_schedule")
    machine = module.ICDORAdaptiveSchedule(pilot=False)
    bad = _v5_ready()
    bad["factor_audit_complete"] = False
    for epoch in range(8):
        machine.update(
            epoch=epoch,
            train_audit_metrics=bad,
            train_calib_metrics=_calib_ready(),
            train_core_metrics=_core_ready(),
        )
    assert machine.failed_closed is False
    assert machine.state == "ADMISSION_CONSOLIDATION"
    assert machine.policy().route_mode == "admitted"


def test_v5_schedule_requires_finite_train_calib_before_route_admission() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_adaptive_schedule")
    machine = module.ICDORAdaptiveSchedule(pilot=True)
    bad_calib = {"source_split": "train_calib", "finite": False}
    for epoch in range(8):
        machine.update(
            epoch=epoch,
            train_audit_metrics=_v5_ready(),
            train_calib_metrics=bad_calib,
            train_core_metrics=_core_ready(),
        )
    assert machine.state == "JOINT_SHADOW"
    assert machine.failed_closed is False
    assert machine.history[-1]["ready"] is False
    assert machine.policy().route_mode == "shadow"


def test_v5_shadow_cannot_transition_without_exact_visual_action_contract() -> None:
    module = importlib.import_module("fate_oia.engine.mosaic_icdor_adaptive_schedule")
    machine = module.ICDORAdaptiveSchedule(pilot=True)
    incomplete = _v5_ready()
    incomplete["final_action_visual_exact"] = False
    for epoch in range(8):
        machine.update(
            epoch=epoch,
            train_audit_metrics=incomplete,
            train_calib_metrics=_calib_ready(),
            train_core_metrics=_core_ready(),
        )
    assert machine.state == "JOINT_SHADOW"
