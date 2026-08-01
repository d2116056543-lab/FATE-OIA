import torch

from fate_oia.engine.train_acpr_meter_oia import (
    _action_credit_parameters,
    _state_measurement_parameters,
)
from fate_oia.models.meter_oia_model import METEROIAModel


def test_gradient_probe_covers_every_action_credit_owner_and_action_state_reader() -> None:
    model = METEROIAModel(dim=384, use_mock_dino=True)
    action_credit = {id(parameter) for parameter in _action_credit_parameters(model)}
    state_measurement = {id(parameter) for parameter in _state_measurement_parameters(model)}
    expected_action_credit = (
        tuple(model.heca_adapters.action_private_adapter.parameters())
        + tuple(model.typed_factors.action_bridge_proj.parameters())
        + tuple(model.action_transport.parameters())
    )
    expected_state_measurement = (
        (model.typed_factors.state_weight, model.typed_factors.state_bias)
        + (model.typed_factors.action_state_embeddings,)
        + tuple(model.typed_factors.state_text_proj.parameters())
        + tuple(model.typed_factors.global_proj.parameters())
    )
    assert {id(parameter) for parameter in expected_action_credit} <= action_credit
    assert {id(parameter) for parameter in expected_state_measurement} <= state_measurement
