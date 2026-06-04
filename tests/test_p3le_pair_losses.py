import torch

from fate_oia.losses.p3le_pair_losses import p3le_pair_loss
from fate_oia.models.p3le_pair_oia_model import P3LEPairOIAFeatureModel


class Args:
    asl_gamma_pos = 0.0
    asl_gamma_neg = 4.0
    asl_clip = 0.05
    loss_action_gt = 1.0
    loss_reason_gt = 1.0
    loss_a_action = 0.5
    loss_r_reason = 0.5
    loss_a_reason = 0.05
    loss_r_action = 0.0
    loss_action_set = 0.1
    loss_pair_seed = 0.05
    loss_pair_consistency = 0.02
    loss_evidence_bag = 0.01
    loss_q_entropy = 0.001
    loss_pareto = 0.1
    pareto_margin_action = 0.005
    pareto_margin_reason = 0.005


def test_pair_loss_has_gradients_and_q_weighted_reason_terms():
    model = P3LEPairOIAFeatureModel(dim=32, action_dim=4, reason_dim=21)
    action = torch.randint(0, 2, (2, 4)).float()
    reason = torch.randint(0, 2, (2, 21)).float()
    out = model(torch.randn(2, 17, 32), action, reason, epoch=12)
    loss, parts = p3le_pair_loss(out, action, reason, Args())
    loss.backward()
    assert float(loss.detach()) > 0
    assert "pair_seed_loss" in parts
    assert "q_mean" in parts
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.router.parameters())
