import torch
from fate_oia.models.acpr_ntmcal_reason_residual import NativeTextReasonResidual

def test_reason_residual_cap():
    m = NativeTextReasonResidual()
    out0 = m(torch.zeros(2,21), torch.randn(2,21,384), torch.rand(2,21), torch.rand(2,21), torch.rand(2,21), epoch=0)
    assert out0["reason_delta"].abs().max() == 0
    out = m(torch.zeros(2,21), torch.randn(2,21,384), torch.rand(2,21), torch.rand(2,21), torch.rand(2,21), epoch=8)
    assert out["reason_delta"].abs().max() <= 0.1801
