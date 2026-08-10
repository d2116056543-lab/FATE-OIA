import numpy as np

from fate_oia.engine.evaluate_dice_oia_probe import SCALAR_METRICS, _scalar_metric, _thresholded_rows


def _rows():
    return {
        "action": np.array([[2.0, -2.0, 1.0, -1.0], [-1.0, 1.0, -2.0, 2.0]], dtype=np.float32),
        "reason": np.zeros((2, 21), dtype=np.float32),
        "action_target": np.array([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=np.float32),
        "reason_target": np.zeros((2, 21), dtype=np.float32),
    }


def test_bootstrap_metric_contains_scalars_only():
    metrics = _scalar_metric(_rows())
    assert tuple(metrics) == SCALAR_METRICS
    assert all(np.isscalar(value) for value in metrics.values())


def test_threshold_view_changes_logits_without_mutating_source():
    rows = _rows()
    original = rows["action"].copy()
    shifted = _thresholded_rows(rows, np.full(25, 0.7, dtype=np.float32))
    assert np.array_equal(rows["action"], original)
    assert not np.array_equal(shifted["action"], original)
