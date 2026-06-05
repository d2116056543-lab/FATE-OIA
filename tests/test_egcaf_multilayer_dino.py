import torch
from fate_oia.models.egcaf_dino_multilayer import EGCafDinoMultiLayerExtractor


def test_lightweight_multilayer_dino_shapes():
    ext = EGCafDinoMultiLayerExtractor(lightweight=True, embed_dim=32, hook_layers=[3,6,9,12])
    out = ext(torch.randn(2,3,64,96))
    assert out["grid_hw"] == (8,12)
    assert set(out["layer_tokens"]) == {"layer_3","layer_6","layer_9","layer_12"}
    assert out["layer_tokens"]["layer_3"].shape == (2,96,32)
