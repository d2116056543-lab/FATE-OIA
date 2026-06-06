import torch
from torch import nn
from fate_oia.models.diva_caf_oia_model import DIVACAFOIAModel

class MockExtractor(nn.Module):
    def forward(self, images):
        b = images.shape[0]
        toks = {3: torch.randn(b,3600,64), 6: torch.randn(b,3600,64), 9: torch.randn(b,3600,64), 12: torch.randn(b,3600,64)}
        maps = {k: v.transpose(1,2).reshape(b,64,45,80) for k,v in toks.items()}
        return {'tokens_by_layer': toks, 'maps_by_layer': maps, 'patch_hw': (45,80)}

def test_model_forward_train_and_eval_required_outputs():
    model = DIVACAFOIAModel(dim=64, action_dim=4, reason_dim=21, dino_extractor=MockExtractor())
    images = torch.randn(2,3,360,640)
    labels = {'action': torch.randint(0,2,(2,4)).float(), 'reason': torch.randint(0,2,(2,21)).float()}
    out = model(images=images, labels=labels, train_mode=True, scene_state_proxy=torch.ones(2,6))
    required = ['z_fate_action_logits','z_eva_action_logits','z_actor_action_logits','guarded_action_logits','base_reason_logits','reason_factor_logits','final_reason_logits','visual_gate','gate_target','action_evidence_tokens','selected_factor_indices','selected_factor_weights','factor_group_scores','reason_to_factor_attention','selected_vs_random_stats']
    for key in required:
        assert key in out
    eval_out = model(images=images, labels=None, train_mode=False)
    assert eval_out['guarded_action_logits'].shape == (2,4)
    assert eval_out['final_reason_logits'].shape == (2,21)
