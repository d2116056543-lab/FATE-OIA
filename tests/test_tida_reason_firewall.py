import torch

from fate_oia.models.tida_reason_reader import TIDAReasonReader


def test_reason_reader_detaches_shared_inputs():
    module = TIDAReasonReader(dim=8, num_reasons=21)
    reason_nodes = torch.randn(2, 21, 8, requires_grad=True)
    predicate = torch.randn(2, 32, 8, requires_grad=True)
    action = torch.randn(2, 4, 8, requires_grad=True)
    out = module(reason_nodes, predicate, action, torch.ones(2, 36), temporal_scale=1.0)
    out["reason_temporal_delta"].sum().backward()
    assert reason_nodes.grad is not None and reason_nodes.grad.abs().sum() > 0
    assert predicate.grad is None
    assert action.grad is None
