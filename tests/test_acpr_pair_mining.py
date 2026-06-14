import torch

from fate_oia.models.acpr_pair_memory import ACPRPairMemory


def test_acpr_pair_mining_contract():
    m = ACPRPairMemory()
    emb = torch.randn(4, 384)
    action = torch.tensor([[1,0,0,0],[1,0,0,0],[0,1,0,0],[1,0,0,0]]).float()
    reason = torch.zeros(4, 21)
    reason[0, 1] = reason[1, 1] = 1
    reason[3, 2] = 1
    pairs = m.mine_pairs(emb, action, reason, [2])
    assert "positive_pairs" in pairs and "contrast_pairs" in pairs
