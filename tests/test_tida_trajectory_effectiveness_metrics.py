import torch
import pytest

from fate_oia.engine.evaluate_tida_oia import trajectory_traffic_effectiveness_metrics


def test_trajectory_metrics_expose_dynamic_gain_transport_and_grounding_quality():
    target = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0]] * 8)
    sign = 2.0 * target - 1.0
    semantic = 0.15 * sign
    delta = 0.10 * sign
    rows = {
        "semantic_action": semantic,
        "trajectory_action": semantic + delta,
        "video_action": semantic + delta,
        "image_reason": torch.zeros(16, 21),
        "video_reason": torch.zeros(16, 21),
        "action_target": target,
        "reason_target": torch.zeros(16, 21),
        "traffic_trajectory_delta": delta,
        "traffic_trajectory_support": torch.full((16, 4), 0.8),
        "trajectory_attention": torch.full((16, 4, 3), 1.0 / 3.0),
        "trajectory_speed": torch.ones(16, 4, 3, 4),
        "trajectory_cycle_confidence": torch.full((16, 4, 3, 5), 0.9),
        "trajectory_exclusive_displacement": torch.ones(16, 4, 3, 4, 2) * 0.1,
    }
    metrics = trajectory_traffic_effectiveness_metrics(rows)
    assert metrics["target_transport"]["action_signed_margin_mean"] > 0
    assert metrics["target_transport"]["correction_to_harm_ratio"] > 1
    assert metrics["grounding_quality"]["cycle_confidence_mean"] == pytest.approx(0.9)
    assert metrics["dynamic_conditioned"]["high_motion"]["count"] > 0
