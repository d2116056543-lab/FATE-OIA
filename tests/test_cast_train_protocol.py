import inspect
from pathlib import Path

import yaml

import fate_oia.engine.supervise_cast_oia_foreground as supervisor
import fate_oia.engine.train_cast_oia as train_cast


def test_config_protocol_values():
    cfg = yaml.safe_load(Path("configs/fate_oia_train_360x640_cast_oia_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["training"]["batch_size"] == 5
    assert cfg["training"]["gradient_accumulation_steps"] == 6
    assert cfg["training"]["reference_effective_batch"] == 32
    assert cfg["training"]["warmup_epochs"] == 3
    assert cfg["model"]["token_compression"] == "none"
    assert cfg["model"]["feature_cache_enabled"] is False
    assert cfg["data"]["eval_splits"] == "test"
    assert cfg["training"]["best_selection_split"] == "test"


def test_foreground_supervisor_forbidden_patterns_absent():
    src = inspect.getsource(supervisor)
    forbidden = ["Start-Process", "Start-Job", "nohup", "Register-ScheduledTask"]
    assert not any(x in src for x in forbidden)
    assert "FALLBACK_LADDER" in src
    assert "(5, 6)" in src and "(2, 16)" in src


def test_train_protocol_uses_direct_image_test_only_no_cache():
    src = inspect.getsource(train_cast)
    assert "BDDOIAMultiTaskDataset" in src
    assert "load_image=True" in src
    assert "eval_splits" in src
    assert "feature_cache_enabled" in src
    assert "token_compression" in src
    assert "checkpoint_best_test.pth" in src


def test_supervisor_requires_review_pass_bound_to_current_head():
    src = inspect.getsource(supervisor)
    assert "cast_oia_v1_preflight_postcommit" in src
    assert "git rev-parse HEAD" in src
    assert "REVIEW_PASS_CAST_OIA_V1.txt" in src
