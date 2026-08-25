import torch

from fate_oia.engine.evaluate_tida_oia import branch_metrics
from fate_oia.engine.train_tida_oia import (
    apply_locked_image_thresholds,
    train_locked_deployment_views,
)


def test_stable_video_deploy_uses_image_train_calib_thresholds():
    rows = {
        "image_action": torch.tensor([[2.0, -2.0, 1.0, -1.0], [-1.0, 1.0, -2.0, 2.0]]),
        "video_action": torch.tensor([[2.2, -1.8, 1.2, -0.8], [-0.8, 1.2, -1.8, 2.2]]),
        "image_reason": torch.zeros(2, 21),
        "video_reason": torch.full((2, 21), 0.1),
        "action_target": torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        "reason_target": torch.stack((torch.ones(21), torch.zeros(21))),
    }
    thresholds = {"image": torch.full((25,), 0.8), "video": torch.full((25,), 0.2)}

    deploy = train_locked_deployment_views(rows, thresholds)

    expected = branch_metrics(rows, thresholds["image"])["video"]
    assert deploy["video_stable"] == expected
    assert deploy["video_adaptive"] != deploy["video_stable"]


def test_locked_thresholds_require_train_calib_provenance_and_preserve_fitted_diagnostic():
    fitted = {"image": torch.full((25,), 0.4), "video": torch.full((25,), 0.6)}
    locked = [0.55] * 25

    resolved = apply_locked_image_thresholds(
        fitted, locked, source="v7_1_train_calib"
    )

    assert torch.equal(resolved["image"], torch.full((25,), 0.55))
    assert torch.equal(resolved["image_fitted_diagnostic"], fitted["image"])
    try:
        apply_locked_image_thresholds(fitted, locked, source="test_oracle")
    except ValueError:
        pass
    else:
        raise AssertionError("test-derived threshold provenance was accepted")
