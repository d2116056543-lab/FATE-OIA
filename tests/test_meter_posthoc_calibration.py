import torch

from fate_oia.utils.meter_posthoc_calibration import apply_meter_deploy, fit_train_calib_deploy_theta, guard_train_calib_deploy_theta, METERCalibrationResult


def test_calibration_is_posthoc_and_subtractive() -> None:
    logits = torch.randn(10, 4)
    labels = torch.randint(0, 2, (10, 4)).float()
    result = fit_train_calib_deploy_theta(logits, labels, model_state_hash="state")
    assert result.fit_split == "train_calib"
    assert result.representation_updated is False
    assert torch.allclose(apply_meter_deploy(logits, result), logits - result.theta)


def test_calibration_guard_falls_back_on_train_calib_degradation() -> None:
    action_logits = torch.tensor([[5.0], [-5.0], [5.0], [-5.0]])
    reason_logits = torch.tensor([[5.0], [-5.0], [5.0], [-5.0]])
    labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    candidate = METERCalibrationResult(
        theta=torch.tensor([-10.0, -10.0]),
        model_state_hash_before="state",
        model_state_hash_after="state",
        fit_split="train_calib",
        representation_updated=False,
    )
    guarded = guard_train_calib_deploy_theta(action_logits, labels, reason_logits, labels, candidate)
    assert guarded.accepted is False
    assert guarded.fallback_reason == "train_calib_deploy_joint_degradation"
    assert torch.equal(guarded.theta, torch.zeros_like(candidate.theta))
    assert guarded.train_calib_deploy_joint < guarded.train_calib_raw_joint
