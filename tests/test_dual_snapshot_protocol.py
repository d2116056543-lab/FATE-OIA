from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_two_stage_configs_lock_validated_trajectory():
    base = yaml.safe_load((ROOT / "configs/fate_oia_train_360x640_dual_snapshot_oia_v1.yaml").read_text())
    consolidation = yaml.safe_load((ROOT / "configs/fate_oia_train_360x640_dual_snapshot_oia_v1_consolidation.yaml").read_text())

    assert base["training"]["epochs"] == 20
    assert base["training"]["batch_size"] == 6
    assert base["training"]["gradient_accumulation_steps"] == 5
    assert base["training"]["lr_primary"] == 2e-4
    assert consolidation["training"]["epochs"] == 3
    assert consolidation["training"]["lr_primary"] == 2e-5
    assert consolidation["training"]["lr_reason_private"] == 6e-5
    assert consolidation["evidence"]["action_scale_start"] == 1.0
    assert consolidation["reason_private"]["reason_scale_start"] == 0.6
    for config in (base, consolidation):
        assert config["experiment"]["feature_cache_enabled"] is False
        assert config["experiment"]["token_compression"] == "none"
        assert config["data"]["num_workers"] == 8


def test_supervisor_locks_dual_snapshot_deploy_contract():
    source = (ROOT / "scripts/FATE_OIA_dual_snapshot_oia_v1_background.ps1").read_text()
    assert '"--action-late-weight", "0.65"' in source
    assert '"--reason-late-weight", "0.875"' in source
    assert '"--action-shrinkage", "50"' in source
    assert '"--reason-shrinkage", "0"' in source
    assert "--init-model-checkpoint" in source
    assert "checkpoint_best_test_deploy_joint.pth" in source
