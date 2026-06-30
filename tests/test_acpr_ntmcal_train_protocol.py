from pathlib import Path


def test_train_protocol_forbids_val_and_cache_and_writes_required_artifacts():
    txt = Path("fate_oia/engine/train_acpr_ntmcal_oia.py").read_text(encoding="utf-8")
    assert "checkpoint_best_val" not in txt
    assert 'split="test"' in txt
    assert "no_feature_cache" in txt
    assert "require_no_token_compression" in txt
    assert "token_compression none" in txt
    assert "update_train_calib_teacher" in txt
    assert "metrics_deploy_fixed.json" in txt
    assert "metrics_base_fixed.json" in txt
    assert "metrics_oracle_diagnostic.json" in txt
    assert "predicate_attention_mass_sample.pt" in txt
    assert "supervisor_live_status.json" in txt
