import torch

from fate_oia.engine.export_tida_traffic_trajectories import (
    _draw_transport_summary,
    trajectory_case_trace,
)


def test_trajectory_case_trace_contains_exact_target_transport_and_tracks():
    output = {
        "image_action_logits": torch.zeros(1, 4),
        "traffic_trajectory_delta": torch.tensor([[0.1, -0.1, 0.0, 0.05]]),
        "video_action_logits": torch.tensor([[0.1, -0.1, 0.0, 0.05]]),
        "traffic_trajectory_trust": torch.full((1, 4), 0.2),
        "traffic_trajectory_support": torch.full((1, 4), 0.8),
        "trajectory_attention": torch.full((1, 4, 2), 0.5),
        "trajectory_cycle_confidence": torch.full((1, 4, 2, 3), 0.9),
        "trajectory_xy": torch.zeros(1, 4, 2, 3, 2),
        "trajectory_speed": torch.ones(1, 4, 2, 2),
        "trajectory_acceleration": torch.zeros(1, 4, 2, 2),
        "trajectory_radial_motion": torch.zeros(1, 4, 2, 2),
    }
    trace = trajectory_case_trace(output, 0, "clip.mp4")
    assert trace["file_name"] == "clip.mp4"
    assert len(trace["actions"]) == 4
    assert len(trace["actions"][0]["tracks"]) == 2
    reconstructed = torch.tensor(trace["image_action_logits"]) + torch.tensor(trace["trajectory_delta"])
    torch.testing.assert_close(reconstructed, torch.tensor(trace["trajectory_action_logits"]))


def test_transport_summary_visualizes_order_state_and_total_credit(tmp_path):
    output = {
        "traffic_trajectory_order_delta": torch.tensor([[0.01, -0.02, 0.03, -0.04]]),
        "traffic_trajectory_state_effective_delta": torch.tensor([[0.02, -0.01, 0.01, -0.02]]),
        "traffic_trajectory_delta": torch.tensor([[0.03, -0.03, 0.04, -0.06]]),
        "traffic_trajectory_utility_gate": torch.tensor([[0.8, 0.6, 0.7, 0.5]]),
        "traffic_trajectory_state_utility_gate": torch.tensor([[0.2, 0.1, 0.3, 0.1]]),
        "trajectory_interaction_risk": torch.full((1, 4, 3), 0.25),
    }
    target = tmp_path / "traffic_credit.png"
    _draw_transport_summary(output, 0, target)
    assert target.exists() and target.stat().st_size > 0
