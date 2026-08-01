import torch

from fate_oia.models.meter_oia_model import METEROIAModel


def _grad_sum(parameters) -> float:
    return sum(
        float(parameter.grad.detach().abs().sum())
        for parameter in parameters
        if parameter.grad is not None
    )


def test_pu_logits_update_reason_private_parameters_only() -> None:
    model = METEROIAModel(dim=384, use_mock_dino=True, state_effect_rank=64)
    output = model(torch.randn(2, 3, 360, 640), progress=0.5)
    output["reason_logits_pu_private"].sum().backward()

    assert _grad_sum(model.heca_adapters.reason_private_adapter.parameters()) > 0.0
    assert _grad_sum(model.heca_adapters.pu_private_head.parameters()) > 0.0
    assert _grad_sum(model.heca_adapters.shared_adapter.parameters()) == 0.0
    assert _grad_sum(model.typed_factors.parameters()) == 0.0
    assert _grad_sum(model.heca_adapters.action_private_adapter.parameters()) == 0.0
    assert _grad_sum(model.typed_factors.action_bridge_proj.parameters()) == 0.0
    assert _grad_sum(model.action_transport.parameters()) == 0.0
    assert _grad_sum(model.foundation.parameters()) == 0.0


def test_state_effect_rank_is_not_ignored() -> None:
    model = METEROIAModel(dim=384, use_mock_dino=True, state_effect_rank=64)
    assert model.action_transport.rank == 64
    assert model.action_transport.state_effect_embedding.shape[-1] == 64
