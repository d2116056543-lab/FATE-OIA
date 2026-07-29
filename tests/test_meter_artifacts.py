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
            "typed_evidence": {
                "state_confusion_matrix": [
                    [[0] * 3 for _ in range(3)] for _ in range(21)
                ],
                "source_coverage": [0] * 21,
                "same_type_margin": [None] * 21,
                "mirror_equivariance": [None] * 21,
                "identity_target_delta": [0.0] * 4,
                "identity_wrong_delta": [0.0] * 4,
                "identity_ap_delta_matrix": [
                    [0.0] * 4 for _ in range(4)
                ],
                "factor_off_delta": [0.0] * 4,
                "state_off_delta": [0.0] * 4,
                "cross_sample_swap_effect": [0.0] * 4,
                "train_audit": {"per_factor": [{} for _ in range(21)]},
                "patch_audit": {
                    "unique_sample_count": 2,
                    "action_coverage": [0, 1, 2, 3],
                    "factor_coverage": list(range(12)),
                    "eligible_factor_coverage": list(range(12)),
                    "requested_factor_coverage": list(range(12)),
                    "executed_factor_coverage": list(range(12)),
                    "model_top_factor_coverage": list(range(12)),
                    "selected_minus_control_ci": {
                        "mean": 0.01,
                        "low": 0.001,
                        "high": 0.02,
                        "n_bootstrap": 100,
                        "cluster_count": 2,
                    },
                },
            },
            "pu_stats": {},
            "calibration": {
                "theta": [0.0] * 25,
                "temperature": None,
                "strategy": "global",
                "accepted": True,
                "fallback_reason": "",
                "fit_split": "train_calib",
                "representation_updated": False,
                "train_calib_raw_joint": 0.0,
                "train_calib_deploy_joint": 0.0,
            },
            "runtime": {
                "epoch": 0,
                "train_rows": 1,
                "mean_data_time": 0.1,
                "mean_dino_time": 0.1,
                "dino_call_count": {"main": 1},
                "eval_mode_time": {},
                "peak_reserved_gb": 0.0,
            },
        },
        file_names=["a.jpg", "b.jpg"],
    )
    assert validate_epoch_artifacts(tmp_path / "epoch_000") == []
