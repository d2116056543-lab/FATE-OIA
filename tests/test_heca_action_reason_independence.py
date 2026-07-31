import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def test_reason_private_parameters_do_not_control_action_logits() -> None:
    model = METEROIAModel(dim=384, use_mock_dino=True)
    out = model(torch.randn(1, 3, 360, 640), progress=1.0)
    grads = torch.autograd.grad(
        out["action_logits_final"].sum(), tuple(model.reason_decoder.parameters()), allow_unused=True
    )
    assert all(value is None or value.count_nonzero() == 0 for value in grads)

