import torch

from fate_oia.utils.meter_posthoc_calibration import apply_meter_deploy, fit_train_calib_deploy_theta
from fate_oia.utils.meter_runtime import METERRuntimeProfile, choose_meter_profile


def test_posthoc_calibration_is_train_calib_only_and_preserves_model_hash() -> None:
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8, 4)).float()
    result = fit_train_calib_deploy_theta(logits, labels, model_state_hash="abc")
    assert result.fit_split == "train_calib"
    assert result.representation_updated is False
    assert result.model_state_hash_before == result.model_state_hash_after == "abc"
    assert torch.allclose(apply_meter_deploy(logits, result), logits - result.theta)


def test_runtime_selection_respects_hard_memory_limit() -> None:
    profiles = [
        METERRuntimeProfile(16, 2, reserved_gb=44.0, samples_per_sec=100),
        METERRuntimeProfile(12, 3, reserved_gb=41.0, samples_per_sec=95),
        METERRuntimeProfile(8, 4, reserved_gb=30.0, samples_per_sec=80),
    ]
    result = choose_meter_profile(profiles)
    assert result.reserved_gb < 45.0
    assert result.effective_batch == 36
