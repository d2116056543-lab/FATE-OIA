import torch
from fate_oia.models.caf_bilevel_routing import BiLevelFactorRouter
from fate_oia.models.caf_sparse_selection import sparsemax

def test_bilevel_routing_topk_per_action_and_sparsemax():
    assert torch.allclose(sparsemax(torch.tensor([[1.0,1.0]]), dim=-1), torch.tensor([[0.5,0.5]]))
    router = BiLevelFactorRouter(dim=32, action_dim=4, factor_topk=3, group_topk=2)
    action_tokens = torch.randn(2,4,32)
    factors = torch.randn(2,20,32)
    groups = torch.randint(0,7,(2,20))
    out = router(action_tokens, factors, groups)
    assert out['selected_factor_indices'].shape == (2,4,3)
    assert out['selected_factor_weights'].shape == (2,4,3)
    assert 'visual_scores' in out
