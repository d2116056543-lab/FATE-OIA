from pathlib import Path

from fate_oia.engine.collect_tida_tta_outputs import normalize_requested_splits


def test_tta_uses_train_only_selection_and_canonical_flip():
    source = Path("fate_oia/engine/collect_tida_tta_outputs.py").read_text(encoding="utf-8")
    assert 'canonicalize_horizontal_flip=True' in source
    assert source.count("apply_image_stage_c=False") == 2
    assert '("train_calib", "train_audit", "test")' in source
    assert 'reason_tta": "original_only"' in source


def test_tta_collection_can_run_test_only_without_changing_default_protocol():
    assert normalize_requested_splits(None) == ("train_calib", "train_audit", "test")
    assert normalize_requested_splits(["test"]) == ("test",)


def test_tta_collector_exposes_raw_frame_store_fast_path():
    source = Path("fate_oia/engine/collect_tida_tta_outputs.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--frame-store-root")' in source
    assert '"frame_store_root": args.frame_store_root' in source
