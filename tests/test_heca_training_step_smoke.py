import torch

from fate_oia.engine.train_acpr_meter_oia import _compute_losses, _forward_training_batch
from fate_oia.models.meter_oia_model import METEROIAModel
from fate_oia.utils.meter_config import load_meter_config


def test_heca_real_loss_graph_one_update_is_finite() -> None:
    torch.manual_seed(11)
    model = METEROIAModel(dim=384, use_mock_dino=True)
    images = torch.randn(2, 3, 360, 640)
    output, view, _ = _forward_training_batch(
        model, images, progress=0.1, mirror_due=True, view_kind="mirror"
    )
    output["reason_ema_probability"] = torch.sigmoid(
        output["reason_logits_global"].detach()
    )
    batch = {
        "action": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]),
        "reason": torch.zeros(2, 21),
        "file_name": ["a", "b"],
    }
    total, parts = _compute_losses(
        model,
        output,
        batch,
        config=load_meter_config("configs/fate_oia_train_360x640_acpr_meter_oia_v3_heca.yaml"),
        grounding_ramp=0.5,
        mechanism_ramp=0.5,
        pu_lambda=torch.zeros(21),
        mirror_output=view,
        corruption_step=1,
        view_kind="mirror",
    )
    action_shared = parts["loss_registry"].owner_total({"action"})
    reason_shared = parts["loss_registry"].owner_total({"reason", "reason_private"})
    shared = list(model.heca_adapters.shared_adapter.parameters())
    assert any(value is not None for value in torch.autograd.grad(action_shared, shared, retain_graph=True, allow_unused=True))
    assert any(value is not None for value in torch.autograd.grad(reason_shared, shared, retain_graph=True, allow_unused=True))
    total.backward()
    assert torch.isfinite(total)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
