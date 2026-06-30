import torch
from fate_oia.models.acpr_ntmcal_model import ACPRNTMCalModel

def test_model_forward_mock_shapes_and_action_independence():
    m = ACPRNTMCalModel(use_mock_dino=True, predicate_topk=8)
    x = torch.randn(2,3,360,640); y = torch.zeros(2,21); y[:,0]=1
    out = m(x, epoch=8, reason_labels=y)
    assert out["action_logits_deploy"].shape == (2,4)
    assert out["reason_logits_deploy"].shape == (2,21)
    z = m(x, epoch=8, reason_labels=y, force_zero_reason_delta=True)
    assert (out["action_logits_ntmcal"] - z["action_logits_ntmcal"]).abs().max() < 1e-6
