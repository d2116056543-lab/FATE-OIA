from __future__ import annotations

from pathlib import Path

from fate_oia.utils.sure_review_gates import assert_test_only_manifest


def test_train_sure_static_test_only_no_feature_cache() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "fate_oia" / "engine" / "train_sure_oia.py").read_text(encoding="utf-8")
    assert '"test"' in text
    assert "val_loader" not in text
    assert "feature_cache" not in text
    assert ".h5" not in text
    assert_test_only_manifest({"eval_splits": ["test"], "uses_val": False, "uses_feature_cache": False})
