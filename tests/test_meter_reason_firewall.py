import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def test_private_reason_loss_does_not_update_foundation_core() -> None:
    torch.manual_seed(9)
    model = METEROIAModel(use_mock_dino=True)
    images = torch.randn(1, 3, 360, 640)
    output = model(images, progress=1.0)
    loss = output["reason_logits_final"].square().mean()
    loss.backward()
    foundation_grads = [p.grad for p in model.foundation.parameters() if p.requires_grad]
    private_grads = [p.grad for p in model.reason_decoder.parameters() if p.requires_grad]
    shared_grads = [p.grad for p in model.heca_adapters.shared_adapter.parameters() if p.requires_grad]
    leaking = [name for name, parameter in model.foundation.named_parameters() if parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0]
    assert not leaking, leaking
    assert any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in private_grads)
    assert any(grad is not None and float(grad.abs().sum()) > 0.0 for grad in shared_grads)
