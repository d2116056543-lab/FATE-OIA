import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def test_reason_loss_cannot_update_typed_factor_or_action() -> None:
    model = METEROIAModel(use_mock_dino=True)
    out = model(torch.randn(1, 3, 360, 640), progress=1)
    out["reason_logits_final"].sum().backward()
    assert all(parameter.grad is None or parameter.grad.eq(0).all() for parameter in model.typed_factors.parameters())
    assert all(parameter.grad is None or parameter.grad.eq(0).all() for parameter in model.action_transport.parameters())
