import torch

from fate_oia.utils.meter_posthoc_calibration import apply_meter_deploy, fit_train_calib_deploy_theta


def test_calibration_is_posthoc_and_subtractive() -> None:
    logits = torch.randn(10, 4)
    labels = torch.randint(0, 2, (10, 4)).float()
    result = fit_train_calib_deploy_theta(logits, labels, model_state_hash="state")
    assert result.fit_split == "train_calib"
    assert result.representation_updated is False
    assert torch.allclose(apply_meter_deploy(logits, result), logits - result.theta)
