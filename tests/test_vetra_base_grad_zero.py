import torch

from vetra_test_utils import build_model, fake_base


def test_transport_backward_never_populates_base_gradients():
    model = build_model()
    out = model.decode_base_output(fake_base(batch=2), alpha=1.0)
    out["action_logits_final"].sum().backward()
    assert all(parameter.grad is None for parameter in model.base_model.parameters())
    assert any(parameter.grad is not None for parameter in model.transport.parameters())

