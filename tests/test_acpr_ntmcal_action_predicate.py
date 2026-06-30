import torch
from fate_oia.models.acpr_ntmcal_action_predicate_head import NativeTextActionPredicateHead

def test_action_predicate_schedule():
    m = NativeTextActionPredicateHead(40)
    base = torch.zeros(2,4); q = torch.rand(2,40); rho = torch.rand(2,40); tok = torch.randn(2,40,384)
    assert m(base,q,rho,tok,epoch=0)["action_predicate_delta"].abs().sum() == 0
    assert m(base,q,rho,tok,epoch=7)["action_predicate_delta"].shape == (2,4)
