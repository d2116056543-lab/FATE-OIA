import json

import pytest
import torch

from fate_oia.engine.eval_dual_snapshot_oia import (
    DualSnapshotWeights,
    blend_snapshots,
    fit_dual_thresholds,
    load_snapshot_artifacts,
)


def test_task_specific_weights_blend_late_and_early_in_declared_direction():
    early = torch.zeros(2, 3)
    late = torch.ones(2, 3)
    weights = DualSnapshotWeights(action_late=0.65, reason_late=0.875)

    assert torch.allclose(blend_snapshots(early, late, weights.action_late), torch.full((2, 3), 0.65))
    assert torch.allclose(blend_snapshots(early, late, weights.reason_late), torch.full((2, 3), 0.875))


def test_fit_dual_thresholds_uses_independent_task_shrinkage():
    generator = torch.Generator().manual_seed(7)
    early_action = torch.randn(80, 4, generator=generator)
    late_action = early_action + 0.1 * torch.randn(80, 4, generator=generator)
    early_reason = torch.randn(80, 21, generator=generator)
    late_reason = early_reason + 0.1 * torch.randn(80, 21, generator=generator)
    action_target = (early_action > 0).float()
    reason_target = (early_reason > 0.7).float()

    result = fit_dual_thresholds(
        early_action,
        late_action,
        early_reason,
        late_reason,
        action_target,
        reason_target,
        DualSnapshotWeights(action_late=0.65, reason_late=0.875),
        action_shrinkage=50.0,
        reason_shrinkage=0.0,
    )

    assert result["action_thresholds"].shape == (4,)
    assert result["reason_thresholds"].shape == (21,)
    assert result["action_shrinkage"] == 50.0
    assert result["reason_shrinkage"] == 0.0
    assert torch.isfinite(result["action_thresholds"]).all()
    assert torch.isfinite(result["reason_thresholds"]).all()


def _write_epoch(path, names, action_labels):
    path.mkdir()
    torch.save(
        {
            "action_logits": torch.randn(2, 4),
            "reason_logits": torch.randn(2, 21),
            "action_labels": action_labels,
            "reason_labels": torch.zeros(2, 21),
        },
        path / "train_calib_logits.pt",
    )
    torch.save(torch.randn(2, 4), path / "action_logits_final_test.pt")
    torch.save(torch.randn(2, 21), path / "reason_logits_final_test.pt")
    torch.save(action_labels, path / "labels_action_test.pt")
    torch.save(torch.zeros(2, 21), path / "labels_reason_test.pt")
    (path / "file_names_train_calib.json").write_text(json.dumps(names))
    (path / "file_names_test.json").write_text(json.dumps(["t0.jpg", "t1.jpg"]))


def test_load_snapshot_artifacts_rejects_misaligned_calibration_names(tmp_path):
    labels = torch.zeros(2, 4)
    early, late = tmp_path / "early", tmp_path / "late"
    _write_epoch(early, ["a.jpg", "b.jpg"], labels)
    _write_epoch(late, ["b.jpg", "a.jpg"], labels)

    with pytest.raises(ValueError, match="calibration file names"):
        load_snapshot_artifacts(early, late)
