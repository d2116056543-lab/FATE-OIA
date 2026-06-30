from pathlib import Path


def test_train_protocol_forbids_val_and_cache():
    txt = Path("fate_oia/engine/train_acpr_ntmcal_oia.py").read_text(encoding="utf-8")
    assert "checkpoint_best_val" not in txt
    assert 'split="test"' in txt
    assert "no_feature_cache" in txt
