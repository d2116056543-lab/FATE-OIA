import torch

from fate_oia.engine.profile_acpr_meter_oia import build_two_stage_profile_plan
from fate_oia.utils.meter_posthoc_calibration import apply_meter_deploy, fit_train_calib_deploy_theta
from fate_oia.utils.meter_runtime import METERRuntimeProfile, choose_meter_profile


def test_posthoc_calibration_is_train_calib_only_and_preserves_model_hash() -> None:
    logits = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8, 4)).float()
    result = fit_train_calib_deploy_theta(logits, labels, model_state_hash="abc")
    assert result.fit_split == "train_calib"
    assert result.representation_updated is False
    assert result.model_state_hash_before == result.model_state_hash_after == "abc"
    assert result.temperature is not None
    assert torch.allclose(
        apply_meter_deploy(logits, result),
        logits / result.temperature - result.theta,
    )


def test_runtime_selection_respects_hard_memory_limit() -> None:
    profiles = [
        METERRuntimeProfile(16, 2, reserved_gb=44.0, samples_per_sec=100),
        METERRuntimeProfile(12, 3, reserved_gb=41.0, samples_per_sec=95),
        METERRuntimeProfile(8, 4, reserved_gb=30.0, samples_per_sec=80),
    ]
    result = choose_meter_profile(profiles)
    assert result.reserved_gb < 45.0
    assert result.effective_batch == 36


def test_runtime_profiler_uses_two_stage_search_not_cartesian_product() -> None:
    candidates = [(16, 2), (12, 3), (8, 4), (6, 6)]
    plan = build_two_stage_profile_plan(tuple(candidates))
    assert plan["candidates"] == [list(item) for item in candidates]
    assert plan["selection"] == "fastest_stable_below_reserved_limit"
