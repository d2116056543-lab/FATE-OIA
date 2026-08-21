import torch

from fate_oia.models.tida_reason_reader import TIDAReasonReader


def test_private_reason_parameters_receive_gradient():
    module = TIDAReasonReader(dim=8)
    out = module(torch.randn(2, 21, 8), torch.randn(2, 32, 8), torch.randn(2, 4, 8), torch.ones(2, 36), temporal_scale=1.0)
    out["reason_temporal_delta"].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
