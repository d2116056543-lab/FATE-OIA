from pathlib import Path

import numpy as np
import pytest
import torch

from fate_oia.engine.collect_vetra_tta_outputs import remap_action_outputs
from fate_oia.utils.vetra_from_scratch import (
    fit_action_combo_oof,
    fit_label_thresholds,
    fit_stable_label_thresholds,
    select_action_combo_hyperparameters,
    validate_clean_stage1_command,
    validate_internal_stage_checkpoint,
)


def test_clean_stage1_rejects_task_checkpoint_arguments():
    validate_clean_stage1_command(["train_aie_oia", "--config", "clean.yaml"])
    with pytest.raises(ValueError, match="init-model-checkpoint"):
        validate_clean_stage1_command(["train_aie_oia", "--init-model-checkpoint", "old.pth"])
    with pytest.raises(ValueError, match="resume"):
        validate_clean_stage1_command(["train_aie_oia", "--resume", "old.pth"])


def test_stage2_checkpoint_must_come_from_current_stage1(tmp_path: Path):
    stage1 = tmp_path / "stage1_aie"
    stage1.mkdir()
    internal = stage1 / "checkpoint_best_test_deploy_joint.pth"
    internal.write_bytes(b"checkpoint")
    assert validate_internal_stage_checkpoint(tmp_path, internal) == internal.resolve()

    external = tmp_path.parent / "external.pth"
    external.write_bytes(b"external")
    with pytest.raises(ValueError, match="outside current clean run"):
        validate_internal_stage_checkpoint(tmp_path, external)


def test_combo_calibration_and_thresholds_are_fit_from_training_rows():
    logits = np.array(
        [
            [4, -4, -4, -4], [3, -3, -3, -3],
            [-4, 4, -4, -4], [-3, 3, -3, -3],
            [3, -3, 3, -3], [2, -2, 2, -2],
            [3, -3, -3, 3], [2, -2, -2, 2],
        ],
        dtype=np.float64,
    )
    targets = (logits > 0).astype(np.float64)
    fitted = fit_action_combo_oof(logits, targets, regularization_c=1.0, folds=2, seed=7)
    assert fitted["oof_action_probability"].shape == targets.shape
    assert np.isfinite(fitted["oof_action_probability"]).all()
    thresholds = fit_label_thresholds(fitted["oof_action_probability"], targets)
    assert thresholds.shape == (4,)
    assert np.logical_and(thresholds >= 0.01, thresholds <= 0.99).all()


def test_nested_train_only_selection_prefers_informative_original_view():
    rng = np.random.default_rng(17)
    targets = rng.integers(0, 2, size=(160, 4)).astype(np.float64)
    signed = targets * 2.0 - 1.0
    original = 3.0 * signed + rng.normal(0.0, 0.25, size=targets.shape)
    flipped = rng.normal(0.0, 3.0, size=targets.shape)

    selected = select_action_combo_hyperparameters(
        original,
        flipped,
        targets,
        original_weights=(0.0, 0.5, 1.0),
        regularization_cs=(1.0,),
        outer_folds=4,
        inner_folds=3,
        seed=19,
    )

    assert selected["selected_original_weight"] == 1.0
    assert selected["selected_regularization_c"] == 1.0
    assert len(selected["candidate_scores"]) == 3
    assert all(len(row["fold_scores"]) == 4 for row in selected["candidate_scores"])
    assert selected["selection_split"] == "provided_train_rows_nested_oof"


def test_nested_train_only_selection_rejects_empty_candidate_grid():
    logits = np.zeros((12, 4), dtype=np.float64)
    targets = np.zeros_like(logits)
    with pytest.raises(ValueError, match="candidate grid"):
        select_action_combo_hyperparameters(
            logits,
            logits,
            targets,
            original_weights=(),
            regularization_cs=(1.0,),
            outer_folds=3,
            inner_folds=2,
        )


def test_stable_thresholds_are_deterministic_jackknife_medians():
    rng = np.random.default_rng(23)
    probability = rng.uniform(0.0, 1.0, size=(120, 4))
    targets = (probability > np.array([0.35, 0.45, 0.55, 0.65])).astype(np.float64)

    first = fit_stable_label_thresholds(probability, targets, folds=6, seed=29)
    second = fit_stable_label_thresholds(probability, targets, folds=6, seed=29)

    assert first["thresholds"].shape == (4,)
    assert first["fold_thresholds"].shape == (6, 4)
    assert np.allclose(first["thresholds"], second["thresholds"])
    assert np.allclose(first["thresholds"], np.median(first["fold_thresholds"], axis=0))
    assert np.logical_and(first["thresholds"] >= 0.01, first["thresholds"] <= 0.99).all()


def test_horizontal_flip_action_outputs_are_remapped_to_original_semantics():
    outputs = {
        "action_primary": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "action_final": torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
    }
    remapped = remap_action_outputs(outputs)
    assert torch.equal(remapped["action_primary"], torch.tensor([[1.0, 2.0, 4.0, 3.0]]))
    assert torch.equal(remapped["action_final"], torch.tensor([[5.0, 6.0, 8.0, 7.0]]))
