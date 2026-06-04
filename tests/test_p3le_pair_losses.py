import torch

from fate_oia.losses.p3le_pair_losses import p3le_pair_loss
from fate_oia.losses.pcgrad_lite import apply_pcgrad_lite, compute_pcgrad_lite
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
    prior = torch.zeros(2, 21)
    prior[:, [5, 6, 9]] = 1.0
    out = model(torch.randn(2, 17, 32), action, reason, evidence_prior=prior, epoch=12)
    loss, parts, tensors = p3le_pair_loss(out, action, reason, Args(), return_tensors=True)
    loss.backward()
    assert float(loss.detach()) > 0
    assert "pair_seed_loss" in parts
    assert "q_mean" in parts
    assert "bdd100k_prior_positive_rate" in parts
    assert "pair_stage_active" in parts
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.router.parameters())


def test_pcgrad_lite_computes_conflict_stats():
    linear = torch.nn.Linear(3, 1)
    x = torch.randn(4, 3)
    y1 = linear(x).mean()
    y2 = -linear(x).mean()
    projected, stats = compute_pcgrad_lite([y1, y2], linear.parameters(), retain_graph=True)
    assert projected
    assert stats["pcgrad_task_count"] == 2.0
    assert "pcgrad_conflict_count" in stats
    assert "pairwise_negative_dot_count" in stats
    stats2 = apply_pcgrad_lite([y1, y2], linear.parameters(), retain_graph=True)
    assert "projection_applied_count" in stats2
