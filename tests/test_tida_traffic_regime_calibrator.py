import numpy as np

from fate_oia.engine.fit_tida_traffic_regime_calibrator import (
    _best_threshold,
    _cluster_thresholds,
    _predict,
)


def test_best_threshold_moves_boundary_when_training_labels_require_it():
    margin = np.tile(np.array([-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]), 8)
    target = np.tile(np.array([0, 0, 1, 1, 1, 1], dtype=bool), 8)
    assert _best_threshold(margin, target) < 0


def test_cluster_thresholds_are_action_and_regime_specific():
    margin = np.array([[-0.2, 0.2], [0.2, -0.2], [-0.2, 0.2], [0.2, -0.2]])
    target = np.array([[1, 1], [1, 0], [0, 1], [1, 1]], dtype=bool)
    cluster = np.array([0, 0, 1, 1])
    result = _cluster_thresholds(margin, target, cluster, cluster_count=2)
    assert result.shape == (2, 2)


def test_zero_shrink_is_exact_baseline_fallback():
    margin = np.array([[-0.2, 0.2], [0.2, -0.2]])
    cluster = np.array([0, 1])
    thresholds = np.array([[0.3, -0.3], [-0.4, 0.4]])
    prediction, boundary = _predict(margin, cluster, thresholds, shrink=0.0, cap=0.2)
    assert np.array_equal(prediction, margin >= 0)
    assert np.allclose(boundary, 0)
