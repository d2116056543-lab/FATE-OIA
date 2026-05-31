import torch
from fate_oia.models.prototype_calibration import apply_prototype_calibration, fit_classwise_bias_temperature_reliability

def test_calibration_reliability_params():
    logits = torch.randn(8, 21); labels = (torch.rand(8, 21) > 0.8).float()
    p = fit_classwise_bias_temperature_reliability(logits, labels)
    assert "bias" in p and "temperature" in p and "reliability_coef" in p
    assert not torch.allclose(apply_prototype_calibration(logits, p), logits)
