import torch
from torch import nn
from fate_oia.losses.diva_caf_losses import diva_caf_loss
from fate_oia.losses.diva_caf_gradient_budget import apply_gradient_budget

def test_losses_include_main_branches_and_gradient_budget_uses_grad_norm():
    p = nn.Parameter(torch.tensor(1.0))
    out = {
      'z_fate_action_logits': torch.zeros(2,4, requires_grad=True) + p,
      'z_eva_action_logits': torch.zeros(2,4, requires_grad=True) + p,
      'z_actor_action_logits': torch.zeros(2,4, requires_grad=True) + p,
      'final_reason_logits': torch.zeros(2,21, requires_grad=True) + p,
      'visual_gate': torch.full((2,4),0.2, requires_grad=True),
      'gate_target': torch.zeros(2,4),
      'selected_vs_random_stats': {'loss': torch.tensor(0.1, requires_grad=True)}
    }
    y_action = torch.ones(2,4)
    y_reason = torch.zeros(2,21)
    loss, terms = diva_caf_loss(out, y_action, y_reason)
    assert all(k in terms for k in ['loss_action_fate','loss_action_eva','loss_action_actor','loss_reason'])
    scaled, stats = apply_gradient_budget(terms['main_loss'], terms['aux_loss'], [p], rho=0.15)
    assert 'norm_main' in stats and 'norm_aux' in stats and 'budget_scale' in stats
    assert scaled.requires_grad
