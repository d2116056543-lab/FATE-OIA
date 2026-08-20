from pathlib import Path

import numpy as np
import pytest
import torch

from fate_oia.engine.supervise_vetra_replay_from_scratch import (
    build_replay_commands,
    promote_clean_stage_a,
    promote_internal_continuation,
    validate_replay_config,
)
from fate_oia.utils.vetra_stage_contracts import sha256_file
from fate_oia.utils.vetra_from_scratch import fit_prior_anchored_label_thresholds


REASON_THRESHOLD_PRIOR = [
    0.715, 0.655, 0.675, 0.700, 0.680, 0.470, 0.010,
    0.680, 0.495, 0.240, 0.550, 0.390, 0.250, 0.550,
    0.465, 0.625, 0.605, 0.620, 0.660, 0.600, 0.660,
]


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
            "regularization_c": 1.0,
            "folds": 5,
            "select_action_hyperparameters": True,
            "candidate_original_weights": [0.75],
            "candidate_regularization_cs": [0.1, 1.0, 10.0],
            "reason_threshold_mode": "prior_anchored_train_oof",
            "reason_threshold_prior": REASON_THRESHOLD_PRIOR,
            "reason_prior_min_macro_gain": 0.001,
            "reason_prior_alpha_step": 0.05,
            "reason_threshold_folds": 5,
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
        (("stage_c", "regularization_c"), 10.0, "1"),
        (("stage_c", "select_action_hyperparameters"), False, "nested"),
        (("stage_c", "reason_threshold_mode"), "independent", "prior"),
        (("stage_c", "reason_threshold_prior"), [0.5] * 21, "historical"),
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
    assert "--select-hyperparameters" in deploy
    assert deploy[deploy.index("--original-weight") + 1] == "0.75"
    assert deploy[deploy.index("--regularization-c") + 1] == "1.0"
    assert deploy[deploy.index("--reason-threshold-mode") + 1] == "prior_anchored_train_oof"
    prior_start = deploy.index("--reason-threshold-prior") + 1
    assert [float(value) for value in deploy[prior_start : prior_start + 21]] == pytest.approx(
        REASON_THRESHOLD_PRIOR
    )
    assert "test" not in deploy[deploy.index("--fit-splits") + 1 :]


def test_prior_anchored_thresholds_choose_the_smallest_train_only_useful_update():
    rng = np.random.default_rng(7)
    target = np.tile(np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float), (30, 1))
    probability = np.clip(0.25 + 0.5 * target + rng.normal(0, 0.08, target.shape), 0.01, 0.99)
    result = fit_prior_anchored_label_thresholds(
        probability,
        target,
        prior_thresholds=np.asarray([0.30, 0.70]),
        alpha_grid=np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
        folds=5,
        seed=11,
        minimum_macro_gain=0.001,
    )

    assert result["selected_alpha"] > 0.0
    selected = next(row for row in result["candidate_scores"] if row["alpha"] == result["selected_alpha"])
    baseline = result["candidate_scores"][0]
    assert selected["macro_f1"] >= baseline["macro_f1"] + 0.001
    assert selected["overall_f1"] >= baseline["overall_f1"]
    eligible = [
        row["alpha"]
        for row in result["candidate_scores"]
        if row["macro_f1"] >= baseline["macro_f1"] + 0.001
        and row["overall_f1"] >= baseline["overall_f1"]
    ]
    assert result["selected_alpha"] == min(eligible)
    expected = result["selected_alpha"] * result["full_sample_thresholds"] + (
        1.0 - result["selected_alpha"]
    ) * np.asarray([0.30, 0.70])
    assert np.allclose(result["thresholds"], expected)


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
