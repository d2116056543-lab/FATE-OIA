from pathlib import Path

import yaml

from fate_oia.utils.coev_contracts import formal_total_updates


def test_fixed_training_protocol_and_no_cache_compression():
    cfg=yaml.safe_load(Path("configs/coev_oia_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["training"]["epochs"]==18 and cfg["training"]["effective_batch"]==32
    assert cfg["data"]["history_frames"]==8 and cfg["data"]["epoch_budget"]["epoch_size"]==6400
    assert cfg["data"]["epoch_budget"]["missing_history_quota"]==320
    assert cfg["training"]["total_updates"]==formal_total_updates(6400,epochs=18)[1]==3600
    assert cfg["training"]["warmup_updates"]==180 and cfg["training"]["eval_split"]=="test"
    assert cfg["runtime"]["feature_cache_enabled"] is False and cfg["runtime"]["token_compression"]=="none"
    assert cfg["training"]["threshold"]==.5 and cfg["training"]["no_metric_early_stop"] is True
    assert "COEV_EPOCH_BUDGET_AMENDMENT.md" in cfg["experiment"]["spec_sha256"]


def test_preflight_binds_epoch_budget_amendment_and_dino_runtime_source():
    source=Path("fate_oia/utils/coev_preflight.py").read_text(encoding="utf-8")
    assert '"vision_transformer.py"' in source
    assert '"docs/superpowers/plans/2026-09-07-coev-epoch-budget-amendment.md"' in source
    assert "amendment_hash_matches_spec" in source


def test_foreground_supervisor_is_attached_and_has_heartbeat():
    source=Path("fate_oia/utils/coev_supervisor.py").read_text(encoding="utf-8")
    assert "subprocess.Popen" in source and "coev_supervisor_heartbeat" in source and "child_pid" in source
    for forbidden in ("DETACHED_PROCESS","CREATE_NEW_PROCESS_GROUP","start_new_session","Start-Process","TaskScheduler","nohup"):
        assert forbidden not in source


def test_only_test_is_evaluated_and_completion_is_strict():
    source=Path("fate_oia/engine/train_coev_oia.py").read_text(encoding="utf-8")
    assert "evaluate(model, test_loader, device)" in source
    assert 'formal = epochs == cfg["training"]["epochs"] and args.max_train is None and args.max_test is None' in source
    assert "ReduceLROnPlateau" not in source and "test_loader" in source
    assert 'epoch_sampler_stats["pool_covered_after_epoch_rate"]' in source
    assert 'epoch_sampler_stats["exposure_min_after_epoch"]' in source
