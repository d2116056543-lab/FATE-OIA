import torch
from fate_oia.models.egcaf_dense_adapter import EGCafDrivingDenseAdapter


def test_dense_adapter_pyramid_shapes():
    adapter = EGCafDrivingDenseAdapter(input_dim=32, hidden_dim=32, layer_names=["layer_3","layer_6","layer_9","layer_12"])
    toks = {k: torch.randn(2,96,32) for k in ["layer_3","layer_6","layer_9","layer_12"]}
    out = adapter(toks, (8,12))
    assert len(out["pyramid"]) == 4
    assert out["pyramid"][0]["P1"].shape == (2,32,8,12)
    assert out["pyramid"][0]["P2"].shape[-2:] == (4,6)
