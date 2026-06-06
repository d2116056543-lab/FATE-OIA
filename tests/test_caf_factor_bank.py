import torch
from fate_oia.models.caf_factor_bank import CAFFactorBank

def test_factor_bank_works_without_bdd100k_and_with_proxy():
    bank = CAFFactorBank(dim=32, action_dim=4, factors_per_action=3)
    actor_evidence = torch.randn(2,4,3,32)
    maps = torch.randn(2,32,45,80)
    out = bank(actor_evidence, maps, scene_state_proxy=None, train_mode=False)
    assert out['factor_tokens'].shape[0] == 2
    assert out['factor_available_mask'].all()
    proxy = torch.ones(2,6)
    out2 = bank(actor_evidence, maps, scene_state_proxy=proxy, train_mode=True)
    assert out2['factor_tokens'].shape[1] >= out['factor_tokens'].shape[1]
