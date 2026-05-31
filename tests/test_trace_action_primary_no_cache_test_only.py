from pathlib import Path
from fate_oia.engine.train_trace_oia import parse_args, verify_cache_ready


def test_v2_config_forces_no_cache_and_test_only():
    args = parse_args(["--config", "configs/fate_oia_train_360x640_trace_action_primary_v2.yaml", "--output_dir", ".background_runs/test_parse_v2"])
    assert args.config_version == "trace_oia_action_primary_v2_direct_image"
    assert args.feature_cache_enabled is False
    assert args.feature_cache_build_before_training is False
    assert args.feature_cache_required_hit_rate == 0.0
    assert args.best_selection_split == "test"
    assert args.best_selection_metric == "test_action_primary_score"
    status = verify_cache_ready(args, None, {})
    assert status["status"] == "disabled"


def test_active_supervisor_has_no_background_launch_strings():
    text = Path("fate_oia/engine/supervise_trace_oia_foreground.py").read_text(encoding="utf-8") + Path("scripts/FATE_OIA_trace_oia_v1_foreground.ps1").read_text(encoding="utf-8")
    for bad in ["Start-Process", "Start-Job", "nohup", "detached", "hidden", "Win32_Process", "Invoke-WmiMethod"]:
        assert bad not in text
