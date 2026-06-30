import torch
from fate_oia.models.acpr_ntmcal_pair_memory import NativeTextReasonPairMemory

def test_pair_disabled_before_epoch7():
    m = NativeTextReasonPairMemory()
    logits = torch.randn(3,21); y = torch.zeros(3,21); pu = {"hard_negative_mask": torch.zeros(3,21)}
    loss, stats = m.loss(logits,y,pu,0,torch.tensor(1.0))
    assert loss.item() == 0
    assert stats["pair_count_total"] == 0
