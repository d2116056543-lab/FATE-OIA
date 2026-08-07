import torch
from fate_oia.utils.aie_cert_counterfactual import AIECertCounterfactualEngine


def test_robust_certificate_uses_mean_plus_std_and_three_controls():
    original=torch.tensor([1.]); selected=torch.tensor([.4]); controls=torch.tensor([[.9,.8,.7,.6]])
    out=AIECertCounterfactualEngine().summarize(original,selected,controls,torch.tensor([[1,1,1,0]],dtype=torch.bool))
    drops=1-controls[0,:3]; expected=.6-(drops.mean()+drops.std(unbiased=False))
    assert out['valid_mask'].item() and torch.allclose(out['certificate'],expected[None])
