import torch

from fate_oia.models.acpr_calibration import ACPRCalibrationHead


def test_acpr_calibration_clamps():
    h = ACPRCalibrationHead()
    out = h(torch.randn(2, 25))
    assert out["calibrated_logits"].shape == (2, 25)
    assert float(out["temperature"].min()) >= 0.5
    assert float(out["temperature"].max()) <= 3.0
