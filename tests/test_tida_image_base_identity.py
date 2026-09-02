from pathlib import Path

import pytest

from fate_oia.engine.train_tida_oia import validate_image_base_identity
from fate_oia.utils.tida_artifacts import file_sha256


def _config(checkpoint: Path, deployment: Path) -> dict:
    return {
        "image_base": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "stage_c_deployment": str(deployment),
            "stage_c_deployment_sha256": file_sha256(deployment),
            "expected_stage_c_metrics": {"Act_mF1": 0.7311},
        }
    }


def test_image_base_identity_validates_both_artifacts(tmp_path: Path):
    checkpoint = tmp_path / "stage_b.pth"
    deployment = tmp_path / "stage_c.pth"
    checkpoint.write_bytes(b"stage-b")
    deployment.write_bytes(b"stage-c")

    identity = validate_image_base_identity(_config(checkpoint, deployment), checkpoint)

    assert identity["checkpoint_sha256"] == file_sha256(checkpoint)
    assert identity["stage_c_deployment_sha256"] == file_sha256(deployment)
    assert identity["expected_stage_c_metrics"] == {"Act_mF1": 0.7311}


def test_image_base_identity_rejects_wrong_checkpoint_hash(tmp_path: Path):
    checkpoint = tmp_path / "stage_b.pth"
    deployment = tmp_path / "stage_c.pth"
    checkpoint.write_bytes(b"stage-b")
    deployment.write_bytes(b"stage-c")
    config = _config(checkpoint, deployment)
    config["image_base"]["checkpoint_sha256"] = "0" * 64

    with pytest.raises(RuntimeError, match="checkpoint SHA256 mismatch"):
        validate_image_base_identity(config, checkpoint)


def test_image_base_identity_rejects_wrong_stage_c_hash(tmp_path: Path):
    checkpoint = tmp_path / "stage_b.pth"
    deployment = tmp_path / "stage_c.pth"
    checkpoint.write_bytes(b"stage-b")
    deployment.write_bytes(b"stage-c")
    config = _config(checkpoint, deployment)
    config["image_base"]["stage_c_deployment_sha256"] = "f" * 64

    with pytest.raises(RuntimeError, match="Stage-C deployment SHA256 mismatch"):
        validate_image_base_identity(config, checkpoint)
