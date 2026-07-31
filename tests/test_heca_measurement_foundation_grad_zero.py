import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def test_measurement_loss_cannot_update_foundation() -> None:
    model = METEROIAModel(dim=384, use_mock_dino=True)
    out = model(torch.randn(1, 3, 360, 640), progress=0.5)
    loss = out["factor_anchor_map"].square().sum() + out["factor_state_logits"].nan_to_num().square().sum()
    loss.backward()
    assert all(parameter.grad is None or parameter.grad.count_nonzero() == 0 for parameter in model.foundation.parameters())

