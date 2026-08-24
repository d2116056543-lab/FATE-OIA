import torch
import pytest

from fate_oia.engine.evaluate_tida_oia import traffic_adaptive_boundary_effectiveness_metrics


def test_boundary_metrics_attribute_corrective_transport_and_dynamic_slices():
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]] * 4)
    sign = 2.0 * target - 1.0
    base = -0.1 * sign
    adaptive = 0.1 * sign
    rows = {
        "video_action_base": base,
        "video_action": adaptive,
        "video_reason": torch.zeros(8, 21),
        "action_target": target,
        "reason_target": torch.zeros(8, 21),
        "traffic_adaptive_boundary_delta": base - adaptive,
        "trajectory_speed": torch.arange(8.0).view(8, 1, 1, 1).expand(8, 4, 2, 2),
        "trajectory_interaction_risk": torch.arange(8.0).view(8, 1, 1).expand(8, 4, 2),
    }
    metrics = traffic_adaptive_boundary_effectiveness_metrics(rows)
    assert metrics["overall"]["Act_mF1_gain"] > 0
    assert metrics["transport"]["gt_margin_sign_agreement"] == 1.0
    assert metrics["decision_flips"]["fn_to_tp"] == [4, 4, 4, 4]
    assert metrics["transport"]["delta_rms_by_action"] == pytest.approx([0.2] * 4)
    assert metrics["dynamic_conditioned"]["high_motion"]["Act_mF1_gain"] > 0
    assert metrics["dynamic_conditioned"]["high_interaction_risk"]["count"] == 2


def test_boundary_metrics_keep_reason_branch_identical():
    rows = {
        "video_action_base": torch.zeros(4, 4),
        "video_action": torch.zeros(4, 4),
        "video_reason": torch.randn(4, 21),
        "action_target": torch.zeros(4, 4),
        "reason_target": torch.zeros(4, 21),
        "traffic_adaptive_boundary_delta": torch.zeros(4, 4),
        "trajectory_speed": torch.zeros(4, 4, 1, 1),
        "trajectory_interaction_risk": torch.zeros(4, 4, 1),
    }
    metrics = traffic_adaptive_boundary_effectiveness_metrics(rows)
    assert metrics["reason_firewall"]["reason_logit_delta_rms"] == 0.0
