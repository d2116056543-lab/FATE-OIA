import torch
from torch import nn

from fate_oia.models.diva_multilayer_dino import DINOIntermediateExtractor

class MockBackbone(nn.Module):
    embed_dim = 384
    def get_intermediate_layers(self, images, n):
        b = images.shape[0]
        return [torch.randn(b, 3600, 384, device=images.device) for _ in range(n)]

def test_dino_multilayer_shape():
    ext = DINOIntermediateExtractor(MockBackbone(), layer_indices=(3,6,9,12), patch_hw=(45,80), dim=384)
    out = ext(torch.randn(2,3,360,640))
    assert len(out['tokens_by_layer']) == 4
    assert out['patch_hw'] == (45,80)
    for k, tokens in out['tokens_by_layer'].items():
        assert tokens.shape == (2,3600,384)
        assert out['maps_by_layer'][k].shape == (2,384,45,80)
