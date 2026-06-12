import torch

from fate_oia.losses import cast_oia_losses as losses


def test_required_loss_functions_are_callable():
    action_logits = torch.randn(3, 4, requires_grad=True)
    action_targets = torch.tensor([[1, 0, 0, 1], [0, 1, 0, 0], [1, 0, 1, 0]], dtype=torch.float32)
    set_logits = torch.randn(3, 16, requires_grad=True)
    set_probs = torch.softmax(set_logits, dim=-1)
    pair_logits = torch.randn(3, 6, requires_grad=True)
    reason_logits = torch.randn(3, 21, requires_grad=True)
    reason_targets = torch.randint(0, 2, (3, 21)).float()
    reliability = torch.rand(3, 21)
    reason_to_set = torch.randn(3, 21, 16, requires_grad=True)
    text_sim = torch.eye(25)
    label_attn = torch.softmax(torch.randn(3, 25, 16), dim=-1)
    edge_weights = torch.softmax(torch.randn(3, 41, 41), dim=-1)
    total = (
        losses.action_multi_label_asl_loss(action_logits, action_targets)
        + losses.action_set_ce_loss(set_logits, action_targets)
        + losses.cardinality_loss(set_probs, action_targets)
        + losses.drop_add_subset_margin_loss(set_logits, action_targets)
        + losses.pair_compatibility_loss(pair_logits, action_targets)
        + losses.reason_reliability_asl_loss(reason_logits, reason_targets, reliability)
        + losses.tail_same_action_set_ranking_loss(reason_logits, reason_targets, action_targets)
        + losses.reason_to_action_set_alignment_loss(reason_to_set, reason_targets, action_targets)
        + losses.text_evidence_contrast_loss(label_attn, text_sim)
        + losses.graph_sparsity_loss(edge_weights)
        + losses.calibration_regularizer(action_logits, action_targets)
        + losses.evidence_compactness_loss(label_attn)
    )
    total.backward()
    assert torch.isfinite(total)
    assert action_logits.grad is not None
