import torch

from fate_oia.engine.evaluate_tida_stage_c_deploy import (
    evaluate_stage_c_branches,
    load_stage_c_deployment,
)


def _artifact(path):
    torch.save(
        {
            "mean": torch.zeros(4),
            "scale": torch.ones(4),
            "coefficient": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            ),
            "intercept": torch.zeros(2),
            "class_codes": torch.tensor([1, 2]),
            "action_thresholds": torch.full((4,), 0.5),
            "reason_thresholds": torch.tensor([0.4, 0.6]),
            "original_weight": 0.75,
            "source_checkpoint_sha256": "stage-b-hash",
        },
        path,
    )


def _rows(video_delta=0.0):
    image_original = torch.tensor([[1.0, -1.0, 0.2, -0.2], [-0.5, 0.5, 0.1, -0.1]])
    image_flip = torch.tensor([[0.8, -0.8, 0.3, -0.3], [-0.4, 0.4, 0.2, -0.2]])
    delta = torch.full_like(image_original, video_delta)
    reason = torch.tensor([[0.2, -0.2], [0.4, -0.4]])
    return {
        "image_action_original": image_original,
        "image_action_flip": image_flip,
        "action_original": image_original + delta,
        "action_flip": image_flip + delta,
        "image_reason_original": reason,
        "reason_original": reason,
        "action_target": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        "reason_target": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    }


def test_load_stage_c_deployment_preserves_train_only_parameters(tmp_path):
    path = tmp_path / "deploy.pth"
    _artifact(path)
    deployment = load_stage_c_deployment(path, torch.device("cpu"))
    assert deployment.source_checkpoint_sha256 == "stage-b-hash"
    torch.testing.assert_close(deployment.reason_thresholds, torch.tensor([0.4, 0.6]))
    assert deployment.action_calibrator.original_weight == 0.75


def test_zero_temporal_delta_is_exact_stage_c_image_fallback(tmp_path):
    path = tmp_path / "deploy.pth"
    _artifact(path)
    views = evaluate_stage_c_branches(
        _rows(video_delta=0.0), load_stage_c_deployment(path, torch.device("cpu"))
    )
    torch.testing.assert_close(
        views["video"]["action_deploy_logits"],
        views["image"]["action_deploy_logits"],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        views["video"]["reason_deploy_logits"],
        views["image"]["reason_deploy_logits"],
        rtol=0,
        atol=0,
    )


def test_temporal_action_is_applied_before_stage_c_combo_calibration(tmp_path):
    path = tmp_path / "deploy.pth"
    _artifact(path)
    views = evaluate_stage_c_branches(
        _rows(video_delta=0.25), load_stage_c_deployment(path, torch.device("cpu"))
    )
    assert not torch.equal(
        views["video"]["action_deploy_logits"],
        views["image"]["action_deploy_logits"],
    )
