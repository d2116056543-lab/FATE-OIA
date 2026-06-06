import torch
from torch import nn
from fate_oia.models.diva_caf_oia_model import DIVACAFOIAModel

class MockExtractor(nn.Module):
    def forward(self, images):
        b = images.shape[0]
        toks = {3: torch.randn(b,3600,32), 6: torch.randn(b,3600,32), 9: torch.randn(b,3600,32), 12: torch.randn(b,3600,32)}
        maps = {k: v.transpose(1,2).reshape(b,32,45,80) for k,v in toks.items()}
        return {'tokens_by_layer': toks, 'maps_by_layer': maps, 'patch_hw': (45,80)}

def test_eval_forward_does_not_require_scene_state_proxy():
    model = DIVACAFOIAModel(dim=32, action_dim=4, reason_dim=21, dino_extractor=MockExtractor())
    out = model(images=torch.randn(2,3,360,640), train_mode=False)
    assert out['no_test_leakage_assertion']['used_bdd100k_gt_in_test_forward'] is False
