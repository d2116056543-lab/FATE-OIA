import torch

from fate_oia.models.adaptive_calibration import AdaptiveCalibrationHead


def test_adaptive_calibration_shapes_and_gradients():
    head = AdaptiveCalibrationHead(32, 25, mode="instance")
    logits = torch.randn(2, 25, requires_grad=True)
    states = torch.randn(2, 8, 32)
    out = head(logits, states)
    assert out["calibrated_logits"].shape == logits.shape
    assert out["calibration_delta_instance"].shape == logits.shape
    loss = out["calibrated_logits"].sum()
    loss.backward()
    assert logits.grad is not None
