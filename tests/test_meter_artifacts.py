from pathlib import Path

import torch

from fate_oia.utils.meter_artifacts import save_epoch_artifacts, validate_epoch_artifacts


def test_epoch_artifact_writer_validates_metadata_and_tensor_outputs(tmp_path: Path) -> None:
    save_epoch_artifacts(
        tmp_path,
        0,
        metrics_raw={"Act_mF1": 0.1},
        metrics_deploy={"Act_mF1": 0.1},
        branch_metrics={},
        logits={
            "action_final_raw_test": torch.zeros(2, 4),
            "reason_final_raw_test": torch.zeros(2, 21),
            "action_visual_test": torch.zeros(2, 4),
            "reason_global_test": torch.zeros(2, 21),
        },
        labels={"action_test": torch.zeros(2, 4), "reason_test": torch.zeros(2, 21)},
        diagnostics={
            "typed_evidence": {},
            "pu_stats": {},
            "calibration": {},
            "runtime": {},
        },
        file_names=["a.jpg", "b.jpg"],
    )
    assert validate_epoch_artifacts(tmp_path / "epoch_000") == []
