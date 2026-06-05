import torch
from fate_oia.models.egcaf_factor_bank import DrivingFactorCandidateBank


def test_factor_bank_shapes_and_sources():
    bank = DrivingFactorCandidateBank(hidden_dim=32, object_factors=4)
    pyramid = [{"P1": torch.randn(2,32,8,12), "P2": torch.randn(2,32,4,6), "P3": torch.randn(2,32,2,3)} for _ in range(4)]
    out = bank(pyramid)
    f = out["factors"]
    assert f.embeddings.shape[0] == 2
    assert f.region_masks.shape[-2:] == (8,12)
    assert f.boxes.min() >= 0 and f.boxes.max() <= 1
    assert {0,1,2,3}.issubset(set(f.source_ids.unique().tolist()))
