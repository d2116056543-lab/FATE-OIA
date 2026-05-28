import torch

from fate_oia.engine.validate_baselines_oia import compute_cached_metrics


def test_compute_cached_metrics_returns_joint_and_map():
    n = 21
    action_logits = torch.full((n, 4), -5.0)
    reason_logits = torch.full((n, 21), -5.0)
    labels_action = torch.zeros(n, 4)
    labels_reason = torch.zeros(n, 21)
    for i in range(n):
        action_logits[i, i % 4] = 5.0
        labels_action[i, i % 4] = 1
        reason_logits[i, i] = 5.0
        labels_reason[i, i] = 1

    metrics = compute_cached_metrics(action_logits, reason_logits, labels_action, labels_reason)
    assert metrics["joint"] > 0.99
    assert metrics["Act_mF1"] > 0.99
    assert metrics["Exp_mF1"] > 0.99
    assert metrics["Exp_mAP"] > 0.99
