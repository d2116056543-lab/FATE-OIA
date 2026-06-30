import torch
from fate_oia.models.acpr_ntmcal_threshold_head import NativeTextMetricCalibrator
from fate_oia.utils.acpr_ntmcal_tensor_asserts import assert_deploy_equation

def test_threshold_deploy_equation():
    m = NativeTextMetricCalibrator()
    a = torch.randn(2,4); r = torch.randn(2,21); s = torch.rand(2,21); c = torch.rand(2,21); rho = torch.rand(2,21); q = torch.rand(2,40); pr = torch.rand(2,40)
    out = m(a,r,s,c,rho,r,q,pr,epoch=8)
    assert_deploy_equation(a, out["theta_action"], out["action_logits_deploy"], "a")
    assert_deploy_equation(r, out["theta_reason"], out["reason_logits_deploy"], "r")
