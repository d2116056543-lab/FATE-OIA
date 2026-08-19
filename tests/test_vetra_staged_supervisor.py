from pathlib import Path

import pytest

from fate_oia.engine.supervise_vetra_staged_from_scratch import (
    build_stage_commands,
    validate_staged_config,
)


def _config():
    return {
        "experiment": {
            "direct_image": True,
            "feature_cache_enabled": False,
            "token_compression": "none",
            "best_selection_split": "train_audit",
            "publication_eligible": True,
        },
        "backbone": {"freeze_backbone": True, "no_grad_backbone": True},
        "stage_a": {"epochs": 10},
        "stage_b": {"epochs": 3},
        "stage_c": {
            "action_fit_splits": ["train_calib", "train_audit"],
            "reason_fit_splits": ["train_calib"],
        },
    }


def test_staged_config_enforces_clean_lineage_and_independent_reason_fit():
    validate_staged_config(_config())
    for mutation, message in (
        (("experiment", "best_selection_split", "test"), "train_audit"),
        (("experiment", "feature_cache_enabled", True), "cache"),
        (("experiment", "token_compression", "keep_merge"), "compression"),
        (("backbone", "freeze_backbone", False), "frozen"),
        (("stage_c", "reason_fit_splits", ["train_calib", "train_audit"]), "reason"),
    ):
        cfg = _config()
        cfg[mutation[0]][mutation[1]] = mutation[2]
        with pytest.raises(ValueError, match=message):
            validate_staged_config(cfg)


def test_commands_start_stage_a_from_random_heads_and_keep_test_out_of_fits(tmp_path: Path):
    commands = build_stage_commands(
        python="python",
        config=tmp_path / "config.yaml",
        run_root=tmp_path / "run",
        cfg=_config(),
        batch_size=6,
        grad_accum=5,
        num_workers=8,
        device="cuda",
        smoke_limits=None,
    )
    stage_a = commands["stage_a"]
    assert "--init-model-checkpoint" not in stage_a
    assert "--resume" not in stage_a
    assert stage_a[stage_a.index("--epochs") + 1] == "10"
    stage_b = commands["stage_b"]
    assert stage_b[stage_b.index("--epochs") + 1] == "3"
    deploy = commands["deploy"]
    action_start = deploy.index("--fit-splits") + 1
    reason_start = deploy.index("--reason-fit-splits") + 1
    assert deploy[action_start:reason_start - 1] == ["train_calib", "train_audit"]
    assert deploy[reason_start:reason_start + 1] == ["train_calib"]
    assert "test" not in deploy[action_start:]
