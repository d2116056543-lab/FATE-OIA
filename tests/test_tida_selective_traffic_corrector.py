import numpy as np
import torch

from fate_oia.engine.fit_tida_selective_traffic_corrector import (
    _apply_rules,
    _audit_action_gate,
    _correction_mask,
    _features,
    _fit_oof,
    _object_track_features,
    _risk_coverage_curve,
    _select_deployment_routes,
    _shuffle_traffic_features,
    _slice_rows,
)


def test_correction_mask_only_changes_confident_near_boundary_disagreements():
    margin = np.array([0.05, 0.05, 0.50, -0.05])
    probability = np.array([0.1, 0.8, 0.1, 0.9])
    mask = _correction_mask(margin, probability, confidence=0.5, bandwidth=0.2)
    assert mask.tolist() == [True, False, False, True]


def test_apply_rules_preserves_baseline_when_rule_is_closed():
    margin = np.array([[0.2, -0.1], [-0.3, 0.4]])
    probability = np.array([[0.1, 0.9], [0.9, 0.1]])
    rules = [{"confidence": 1.0, "bandwidth": 0.0}] * 2
    prediction, mask = _apply_rules(margin, probability, rules)
    assert np.array_equal(prediction, margin >= 0)
    assert not mask.any()


def test_features_include_global_action_and_traffic_context():
    rows = {
        "traffic_trajectory_order_delta": torch.zeros(3, 4),
        "traffic_trajectory_support": torch.ones(3, 4),
        "trajectory_state_strength": torch.ones(3, 4),
        "trajectory_interaction_risk": torch.zeros(3, 4, 2),
        "traffic_trajectory_state_features": torch.zeros(3, 4, 8),
    }
    result = _features(rows, torch.zeros(3, 4))
    assert result.shape == (3, 60)


def test_oof_accepts_numeric_source_ids_with_numpy_string_types():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(100, 3))
    margin = rng.normal(size=(100, 2))
    target = np.stack((np.arange(100) % 2, (np.arange(100) // 2) % 2), axis=1).astype(bool)
    source_ids = np.arange(100) % 2

    probability, models = _fit_oof(features, margin, target, source_ids, folds=2, seed=7)

    assert probability.shape == (100, 2)
    assert np.isfinite(probability).all()
    assert all(len(group) == 2 for group in models)


def test_oof_can_hold_out_each_source_as_a_domain():
    rng = np.random.default_rng(9)
    features = rng.normal(size=(120, 3))
    margin = rng.normal(size=(120, 2))
    target = np.stack((np.arange(120) % 2, (np.arange(120) // 2) % 2), axis=1).astype(bool)
    source_ids = np.arange(120) % 3

    probability, models = _fit_oof(
        features, margin, target, source_ids, folds=3, seed=9,
        fold_mode="leave_source_out",
    )

    assert np.isfinite(probability).all()
    assert all(len(group) == 3 for group in models)


def test_audit_gate_only_enables_traffic_that_beats_both_references():
    target = np.array([[0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1]], dtype=bool)
    baseline = np.array([[1, 0], [1, 1], [0, 0], [1, 1], [0, 0], [0, 1]], dtype=bool)
    full = target.copy()
    full[:, 1] = np.logical_not(target[:, 1])
    margin_only = baseline.copy()
    mask = full != baseline

    gate, rows = _audit_action_gate(baseline, full, margin_only, mask, target)

    assert gate.tolist() == [False, False]  # action 0 has fewer than four audit corrections
    assert rows[0]["full_f1"] > rows[0]["base_f1"]


def test_object_track_features_preserve_pointwise_motion_and_visibility():
    tracks = torch.zeros(2, 15, 4, 2)
    tracks[0, :, :, 0] = torch.arange(15)[:, None]
    visibility = torch.ones(2, 15, 4, dtype=torch.bool)
    visibility[1, :, 0] = False
    store = {
        "file_names": ["a.jpg", "b.jpg"],
        "tracks_xy": tracks,
        "visibility": visibility,
    }

    result = _object_track_features(store, ["b.jpg", "a.jpg"])

    assert result.shape[0] == 2
    assert np.isfinite(result).all()
    assert not np.array_equal(result[0], result[1])


def test_object_track_features_remove_global_camera_translation():
    base = torch.zeros(1, 15, 4, 2)
    drift = base.clone()
    drift[0, :, :, 0] = torch.arange(15)[:, None] * 0.2
    visibility = torch.ones(1, 15, 4, dtype=torch.bool)
    static_store = {"file_names": ["a.jpg"], "tracks_xy": base, "visibility": visibility}
    drift_store = {"file_names": ["a.jpg"], "tracks_xy": drift, "visibility": visibility}

    static_features = _object_track_features(static_store, ["a.jpg"])
    drift_features = _object_track_features(drift_store, ["a.jpg"])

    assert np.allclose(static_features, drift_features, atol=1e-6)


def test_slice_rows_keeps_nested_policy_rows_aligned():
    rows = {
        "file_names": ["a", "b", "c"],
        "action_target": torch.arange(3)[:, None],
        "_policy": {
            "file_names": ["a", "b", "c"],
            "candidate": torch.arange(3)[:, None],
        },
    }
    result = _slice_rows(rows, 1, 3)
    assert result["file_names"] == ["b", "c"]
    assert result["action_target"].squeeze(1).tolist() == [1, 2]
    assert result["_policy"]["candidate"].squeeze(1).tolist() == [1, 2]


def test_deployment_routes_choose_best_calibration_branch_per_action():
    baseline = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool)
    target = baseline.copy()
    margin = baseline.copy()
    full = baseline.copy()
    baseline[:, 0] = ~target[:, 0]
    full[:, 1] = ~target[:, 1]
    margin[:, 2] = ~target[:, 2]

    routes, rows = _select_deployment_routes(baseline, margin, full, target)

    assert routes == ["margin_only", "baseline", "baseline"]
    assert rows[0]["selected_route"] == "margin_only"


def test_risk_coverage_curve_reports_net_repairs_in_confidence_order():
    baseline = np.array([0, 0, 1, 1], dtype=bool)
    probability = np.array([0.9, 0.8, 0.2, 0.4])
    target = np.array([1, 0, 0, 1], dtype=bool)

    curve = _risk_coverage_curve(baseline, probability, target, points=(0.25, 0.5))

    assert curve[0]["selected_count"] == 1
    assert curve[0]["net_corrected"] == 1
    assert curve[-1]["selected_count"] == 2


def test_shuffle_traffic_features_keeps_margin_prefix_fixed():
    features = np.arange(30).reshape(3, 10)
    shuffled = _shuffle_traffic_features(features, np.array([2, 0, 1]), margin_dim=2)

    assert np.array_equal(shuffled[:, :2], features[:, :2])
    assert np.array_equal(shuffled[:, 2:], features[[2, 0, 1], 2:])
