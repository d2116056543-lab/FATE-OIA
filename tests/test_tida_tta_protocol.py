from pathlib import Path


def test_tta_uses_train_only_selection_and_canonical_flip():
    source = Path("fate_oia/engine/collect_tida_tta_outputs.py").read_text(encoding="utf-8")
    assert 'canonicalize_horizontal_flip=True' in source
    assert '("train_calib", "train_audit", "test")' in source
    assert 'reason_tta": "original_only"' in source
