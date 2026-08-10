import torch

from vetra_test_utils import build_model


def test_formal_forward_invokes_base_exactly_once():
    model = build_model()
    model(torch.randn(2, 3, 8, 8), alpha=1.0)
    assert model.base_model.forward_calls == 1

