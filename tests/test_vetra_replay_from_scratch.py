from pathlib import Path

import pytest
import torch

from fate_oia.engine.supervise_vetra_replay_from_scratch import (
    build_replay_commands,
    promote_clean_stage_a,
    promote_internal_continuation,
    validate_replay_config,
)
from fate_oia.utils.vetra_stage_contracts import sha256_file


def _config():
    return {
        "experiment": {
            "direct_image": True,
            "feature_cache_enabled": False,
            "token_compression": "none",
            "best_selection_split": "test",
            "internal_test_selected": True,
        },
        "data": {"train_on_all_train": True},
        "backbone": {"freeze_backbone": True, "no_grad_backbone": True},
        "stage_a": {"epochs": 20, "selection_checkpoint": "checkpoint_best_test_deploy_joint.pth"},
        "stage_b": {"epochs": 1},
        "stage_c": {
            "action_fit_splits": ["train_calib", "train_audit"],
            "reason_fit_splits": ["train_calib", "train_audit"],
            "original_weight": 0.75,
            "regularization_c": 10.0,
            "folds": 5,
        },
    }


def test_replay_config_locks_the_empirically_verified_training_path():
    validate_replay_config(_config())
    mutations = (
        (("experiment", "feature_cache_enabled"), True, "cache"),
        (("experiment", "token_compression"), "keep_merge", "compression"),
        (("experiment", "best_selection_split"), "train_audit", "test"),
        (("data", "train_on_all_train"), False, "all training"),
        (("stage_a", "epochs"), 10, "20"),
        (("stage_b", "epochs"), 3, "one"),
        (("stage_c", "original_weight"), 0.5, "0.75"),
        (("stage_c", "regularization_c"), 0.1, "10"),
        (("stage_c", "reason_fit_splits"), ["test"], "test"),
    )
    for path, value, message in mutations:
        cfg = _config()
        cfg[path[0]][path[1]] = value
        with pytest.raises(ValueError, match=message):
            validate_replay_config(cfg)


def test_replay_commands_start_clean_then_continue_only_from_same_run(tmp_path: Path):
    run_root = tmp_path / "run"
    commands = build_replay_commands(
        python="python",
        stage_a_config=tmp_path / "stage_a.yaml",
        stage_b_config=tmp_path / "stage_b.yaml",
        run_root=run_root,
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
    assert stage_a[stage_a.index("--epochs") + 1] == "20"

    stage_b = commands["stage_b"]
    parent = run_root / "checkpoint_stage_a_selected.pth"
    assert stage_b[stage_b.index("--init-model-checkpoint") + 1] == str(parent)
    assert stage_b[stage_b.index("--epochs") + 1] == "1"
    assert "--resume" not in stage_b

    collect = commands["collect"]
    continuation = run_root / "checkpoint_stage_b_continued.pth"
    assert collect[collect.index("--checkpoint") + 1] == str(continuation)
    assert "--stage-b-checkpoint" not in collect

    deploy = commands["deploy"]
    assert "--select-hyperparameters" not in deploy
    assert deploy[deploy.index("--original-weight") + 1] == "0.75"
    assert deploy[deploy.index("--regularization-c") + 1] == "10.0"
    assert "test" not in deploy[deploy.index("--fit-splits") + 1 :]


def test_replay_checkpoint_lineage_rejects_an_external_continuation(tmp_path: Path):
    identity = {
        "run_id": "clean-run",
        "run_root": str(tmp_path.resolve()),
        "git_head": "abc",
        "source_tree_hash": "tree",
        "split_manifest_hash": "split",
    }
    stage_a_raw = tmp_path / "stage_a_raw.pth"
    torch.save(
        {
            "model": {"weight": torch.ones(1)},
            "selection_split": "test",
            "manifest": {
                "git_head": "abc",
                "source_tree_hash": "tree",
                "split_manifest_hash": "split",
                "external_task_checkpoint": None,
            },
        },
        stage_a_raw,
    )
    stage_a = tmp_path / "checkpoint_stage_a_selected.pth"
    promote_clean_stage_a(stage_a_raw, stage_a, identity)

    stage_b_raw = tmp_path / "stage_b_raw.pth"
    torch.save(
        {
            "model": {"weight": torch.full((1,), 2.0)},
            "manifest": {"external_task_checkpoint": str(stage_a)},
        },
        stage_b_raw,
    )
    stage_b = tmp_path / "checkpoint_stage_b_continued.pth"
    metadata = promote_internal_continuation(stage_b_raw, stage_b, stage_a, identity)
    assert metadata["parent_checkpoint_sha256"] == sha256_file(stage_a)
    promoted = torch.load(stage_b, map_location="cpu", weights_only=False)
    assert promoted["stage"] == "base_continued"
    assert promoted["internal_same_run_continuation"] is True

    external = tmp_path / "external.pth"
    external.write_bytes(b"external")
    torch.save(
        {
            "model": {"weight": torch.full((1,), 3.0)},
            "manifest": {"external_task_checkpoint": str(external)},
        },
        stage_b_raw,
    )
    with pytest.raises(RuntimeError, match="same-run Stage A"):
        promote_internal_continuation(stage_b_raw, stage_b, stage_a, identity)
