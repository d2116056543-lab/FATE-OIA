import torch
from fate_oia.losses.dice_rank_sketch import DistributionalRankSketch, quantile_rank_preservation_loss, rank_preservation_loss


def test_rank_protection_penalizes_new_inversion():
    target=torch.tensor([[1.],[0.]])
    base=torch.tensor([[1.],[-1.]])
    assert rank_preservation_loss(base,base,target)==0
    assert rank_preservation_loss(-base,base,target)>0


def test_quantile_protection_uses_frozen_base_and_final_sketches():
    target=torch.tensor([[1.],[1.],[0.],[0.]])
    base=torch.tensor([[2.],[1.5],[-1.],[-2.]])
    final_sketch=DistributionalRankSketch(1,4,momentum=0); base_sketch=DistributionalRankSketch(1,4,momentum=0)
    final_sketch.update(base,target,1); base_sketch.update(base,target,1)
    good=quantile_rank_preservation_loss(base,base,target,final_sketch,base_sketch)
    bad=quantile_rank_preservation_loss(-base,base,target,final_sketch,base_sketch)
    assert good<bad
