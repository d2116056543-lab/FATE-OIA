import torch
from fate_oia.models.egcaf_dynamic_selector import ExplanationGuidedDynamicFactorSelector
from fate_oia.models.egcaf_factor_types import FactorBatch


def make_factors():
    b,m,d,h,w = 2,12,32,8,12
    return FactorBatch(torch.randn(b,m,d), torch.rand(b,m,h,w), torch.rand(b,m,4).clamp(0,1), torch.zeros(b,m,dtype=torch.long), torch.randn(b,m,11), torch.rand(b,m), torch.ones(b,m,dtype=torch.bool), {})


def test_selector_uses_lambda_and_selects_k():
    sel = ExplanationGuidedDynamicFactorSelector(hidden_dim=32, k_max=3)
    f = make_factors()
    r = torch.randn(2,21)
    out = sel(f, r, torch.rand(2,21), torch.randn(2,6))
    assert out["selected_indices"].shape == (2,4,3)
    assert out["selected_factors"].embeddings.shape == (2,4,3,32)
    with torch.no_grad():
        sel.lambda_exp_max = 0.0
    out2 = sel(f, r, torch.rand(2,21), torch.randn(2,6))
    assert torch.all(out2["lambda_exp"] == 0)
