import torch

from fate_oia.engine.validate_baselines_oia import compute_cached_metrics


def test_compute_cached_metrics_returns_joint_and_map():
    action_logits = torch.tensor([[5.0, -5.0, -5.0, -5.0], [-5.0, 5.0, -5.0, -5.0]])
    reason_logits = torch.full((2, 21), -4.0)
    reason_logits[0, 0] = 4.0
    reason_logits[1, 1] = 4.0
    labels_action = torch.zeros(2, 4)
    labels_reason = torch.zeros(2, 21)
    labels_action[0, 0] = 1
    labels_action[1, 1] = 1
    labels_reason[0, 0] = 1
    labels_reason[1, 1] = 1

    metrics = compute_cached_metrics(action_logits, reason_logits, labels_action, labels_reason)
    assert metrics["joint"] > 0.99
    assert metrics["Act_mF1"] > 0.99
    assert metrics["Exp_mF1"] > 0.99
    assert metrics["Exp_mAP"] > 0.99
