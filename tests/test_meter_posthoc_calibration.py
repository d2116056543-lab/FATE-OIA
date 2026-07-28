import torch

from fate_oia.utils.meter_posthoc_calibration import apply_meter_deploy, fit_train_calib_deploy_theta, guard_train_calib_deploy_theta, METERCalibrationResult


def test_calibration_is_posthoc_and_subtractive() -> None:
    logits = torch.randn(10, 4)
    labels = torch.randint(0, 2, (10, 4)).float()
    result = fit_train_calib_deploy_theta(logits, labels, model_state_hash="state")
    assert result.fit_split == "train_calib"
    assert result.representation_updated is False
    assert result.temperature is not None
    assert torch.allclose(
        apply_meter_deploy(logits, result),
        logits / result.temperature - result.theta,
    )


def test_calibration_guard_falls_back_on_train_calib_degradation() -> None:
    action_logits = torch.tensor([[5.0], [-0.1], [5.0], [-0.1]])
    reason_logits = torch.tensor([[5.0], [-0.1], [5.0], [-0.1]])
    labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    candidate = METERCalibrationResult(
        theta=torch.tensor([-0.2, -0.2]),
        model_state_hash_before="state",
        model_state_hash_after="state",
        fit_split="train_calib",
        representation_updated=False,
    )
    guarded = guard_train_calib_deploy_theta(action_logits, labels, reason_logits, labels, candidate)
    assert guarded.accepted is False
    assert guarded.fallback_reason == "train_calib_deploy_joint_degradation"
    assert torch.equal(guarded.theta, torch.zeros_like(candidate.theta))
    assert guarded.train_calib_deploy_joint >= guarded.train_calib_raw_joint


def test_calibration_guard_checks_ranking_on_logits_before_sigmoid_saturation() -> None:
    generator = torch.Generator().manual_seed(17)
    action_logits = torch.randn(512, 1, generator=generator) * 15
    labels = torch.randint(0, 2, (512, 1), generator=generator).float()
    reason_logits = action_logits.clone()
    candidate = METERCalibrationResult(
        theta=torch.zeros(2),
        temperature=torch.full((2,), 1.5),
        model_state_hash_before="state",
        model_state_hash_after="state",
        fit_split="train_calib",
        representation_updated=False,
    )

    guarded = guard_train_calib_deploy_theta(
        action_logits,
        labels,
        reason_logits,
        labels,
        candidate,
        fallback_on_deploy_degradation=False,
    )

    assert guarded.map_max_abs_delta == 0.0
