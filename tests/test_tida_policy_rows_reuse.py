import pytest
import torch

from fate_oia.engine.train_tida_oia import (
    object_intent_policy_cohorts_enabled,
    validate_and_slice_policy_rows,
)


def _rows():
    return {
        "action_target": torch.arange(12).reshape(6, 2),
        "reason_target": torch.arange(18).reshape(6, 3),
        "pre_object_intent_action": torch.zeros(6, 2),
        "pre_object_intent_reason": torch.zeros(6, 3),
        "source_batches": ["a", "a", "b", "b", "c", "c"],
        "file_names": [f"f{i}" for i in range(6)],
        "_policy_cohort_sizes": {"train_calib": 2, "train_audit": 2, "train_core": 2},
    }


def test_policy_rows_reuse_recovers_train_calib_prefix():
    rows, calib = validate_and_slice_policy_rows(_rows())
    assert rows["action_target"].shape[0] == 6
    assert calib["action_target"].tolist() == [[0, 1], [2, 3]]
    assert calib["source_batches"] == ["a", "a"]
    assert "_policy_cohort_sizes" not in calib


def test_policy_rows_reuse_rejects_test_cohort():
    rows = _rows()
    rows["_policy_cohort_sizes"]["test"] = 1
    with pytest.raises(ValueError, match="test cohort"):
        validate_and_slice_policy_rows(rows)


def test_policy_rows_reuse_rejects_inconsistent_lengths():
    rows = _rows()
    rows["source_batches"].pop()
    with pytest.raises(ValueError, match="row count"):
        validate_and_slice_policy_rows(rows)


def test_disabled_object_intent_skips_expensive_policy_cohorts():
    class Model:
        object_intent_enabled = False

    config = {
        "deployment": {"object_intent_utility_policy_use_train_audit": True}
    }
    assert object_intent_policy_cohorts_enabled(Model(), config) is False
