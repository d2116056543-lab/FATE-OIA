from pathlib import Path

import yaml

from fate_oia.utils.coev_contracts import formal_total_updates


def test_fixed_training_protocol_and_no_cache_compression():
    cfg=yaml.safe_load(Path("configs/coev_oia_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["training"]["epochs"]==24 and cfg["training"]["effective_batch"]==32
    assert cfg["training"]["total_updates"]==formal_total_updates(15302)[1]==11496
    assert cfg["training"]["warmup_updates"]==575 and cfg["training"]["eval_split"]=="test"
    assert cfg["runtime"]["feature_cache_enabled"] is False and cfg["runtime"]["token_compression"]=="none"
    assert cfg["training"]["threshold"]==.5 and cfg["training"]["no_metric_early_stop"] is True


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
