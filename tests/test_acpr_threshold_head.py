import torch

from fate_oia.models.acpr_threshold_head import ACPRThresholdHead


def test_threshold_head_deploys_base_minus_theta_with_group_shrinkage():
    group_ids = torch.tensor([0, 0, 1, 1] + [2 + (i % 3) for i in range(21)])
    delta_scale = torch.ones(25) * 0.5
    head = ACPRThresholdHead(label_group_ids=group_ids, label_delta_scale=delta_scale)
    action = torch.randn(2, 4)
    reason = torch.randn(2, 21)

    out = head(action, reason)

    theta = head.compose_theta()
    assert out["logits_base"].shape == (2, 25)
    assert out["logits_deploy"].shape == (2, 25)
    assert torch.allclose(out["logits_deploy"], out["logits_base"] - theta.view(1, -1))
    assert out["action_logits_deploy"].shape == (2, 4)
    assert out["reason_logits_deploy"].shape == (2, 21)
    assert out["threshold_prob"][:4].min() >= 0.10
    assert out["threshold_prob"][:4].max() <= 0.90
    assert out["threshold_prob"][4:].min() >= 0.01
    assert out["threshold_prob"][4:].max() <= 0.85


def test_threshold_head_teacher_update_does_not_require_test_thresholds():
    head = ACPRThresholdHead()
    teacher = torch.linspace(-2, 2, 25)
    pred_rate = torch.linspace(0.01, 0.5, 25)

    before = head.compose_theta().detach().clone()
    head.update_teacher(teacher, pred_rate_teacher=pred_rate, ema=0.2, copy_to_params=False)
    after_teacher_only = head.compose_theta().detach().clone()
    assert torch.allclose(before, after_teacher_only)
    assert torch.allclose(head.theta_teacher, teacher * 0.2)
    assert torch.allclose(head.teacher_pred_rate, pred_rate * 0.2)

    head.update_teacher(teacher, pred_rate_teacher=pred_rate, ema=1.0, copy_to_params=True)
    after_copy = head.compose_theta().detach()
    assert torch.mean(torch.abs(after_copy - before)) > 1e-4
