import torch
from fate_oia.models.diva_visual_actor import DIVAVisualActor

def test_visual_actor_outputs_action_logits_and_evidence():
    features = {'P1': torch.randn(2,4,64,45,80), 'P2': torch.randn(2,4,64,23,40), 'P3': torch.randn(2,4,64,12,20)}
    actor = DIVAVisualActor(dim=64, action_dim=4, num_heads=4)
    out = actor(features)
    assert out['z_eva'].shape == (2,4)
    assert out['action_evidence_tokens'].shape[:3] == (2,4,3)
    assert out['evidence_confidence'].shape == (2,4)
