from __future__ import annotations

import pytest

from fate_oia.utils.sure_review_gates import assert_test_only_manifest


def test_manifest_gate_rejects_val() -> None:
    with pytest.raises(RuntimeError):
        assert_test_only_manifest({"eval_splits": ["val", "test"], "uses_val": True, "uses_feature_cache": False})


def test_manifest_gate_rejects_cache() -> None:
    with pytest.raises(RuntimeError):
        assert_test_only_manifest({"eval_splits": ["test"], "uses_val": False, "uses_feature_cache": True})
