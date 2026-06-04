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
    tail_indices = [5, 6, 9, 10, 11, 12, 13, 14]


def test_pair_seed_is_reliability_and_evidence_gate_aware():
    model = P3LEPairOIAFeatureModel(dim=32, action_dim=4, reason_dim=21)
    action = torch.ones(2, 4)
    reason = torch.ones(2, 21)
    outputs = model(torch.randn(2, 17, 32), action, reason, evidence_prior=torch.zeros(2, 21), epoch=12)
    outputs["evidence_lambda_active"] = torch.tensor(0.0)
    _, parts, _ = p3le_pair_loss(outputs, action, reason, Args(), return_tensors=True)
    assert parts["mean_pair_seed_weight"] == 0.0
    assert parts["tail_pair_seed_weight_mean"] == 0.0

