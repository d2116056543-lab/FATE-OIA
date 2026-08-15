from pathlib import Path

import numpy as np
import pytest
import torch

from fate_oia.engine.collect_vetra_tta_outputs import remap_action_outputs
from fate_oia.utils.vetra_from_scratch import (
    fit_action_combo_oof,
    fit_label_thresholds,
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


def test_horizontal_flip_action_outputs_are_remapped_to_original_semantics():
    outputs = {
        "action_primary": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
        "action_final": torch.tensor([[5.0, 6.0, 7.0, 8.0]]),
    }
    remapped = remap_action_outputs(outputs)
    assert torch.equal(remapped["action_primary"], torch.tensor([[1.0, 2.0, 4.0, 3.0]]))
    assert torch.equal(remapped["action_final"], torch.tensor([[5.0, 6.0, 8.0, 7.0]]))
