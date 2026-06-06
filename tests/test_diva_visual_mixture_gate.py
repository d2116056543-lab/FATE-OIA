import torch
import torch.nn.functional as F
from fate_oia.models.diva_visual_mixture_gate import SupervisedVisualMixtureGate

def test_visual_gate_formula_and_target_uses_labels():
    gate = SupervisedVisualMixtureGate(action_dim=4, delta_cap=0.08, gate_margin=0.0)
    z_fate = torch.zeros(3,4)
    z_eva = torch.ones(3,4)
    y = torch.ones(3,4)
    out = gate(z_fate, z_eva, torch.ones(3,4), y_action=y, train_mode=True)
    expected = z_fate + out['visual_gate'] * torch.clamp(z_eva - z_fate, -0.08, 0.08)
    assert torch.allclose(out['z_actor'], expected, atol=1e-6)
    assert out['gate_target'].shape == (3,4)
    assert out['visual_gate'].min() >= 0 and out['visual_gate'].max() <= 1
    assert (out['z_actor'] - z_fate).abs().max() <= 0.08001
