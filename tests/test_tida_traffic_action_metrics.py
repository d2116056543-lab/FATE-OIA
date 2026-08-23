import torch

from fate_oia.engine.evaluate_tida_oia import traffic_action_effectiveness_metrics


def test_traffic_effectiveness_measures_target_transport_and_attention():
    target = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    semantic = torch.zeros(2, 4)
    delta = (2.0 * target - 1.0) * 0.2
    attention = torch.zeros(2, 4, 12)
    attention[:, :, :4] = 0.25
    rows = {
        "image_action": semantic,
        "semantic_action": semantic,
        "traffic_action": delta,
        "video_action": delta,
        "image_reason": torch.zeros(2, 21),
        "semantic_reason": torch.zeros(2, 21),
        "video_reason": torch.zeros(2, 21),
        "action_target": target,
        "reason_target": torch.zeros(2, 21),
        "traffic_action_delta": delta,
        "traffic_motion_energy": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "traffic_action_attention": attention,
        "traffic_same_action_mass": torch.full((2, 4), 0.25),
    }
    result = traffic_action_effectiveness_metrics(rows)
    assert result["target_transport"]["action_signed_margin_mean"] > 0
    assert result["target_transport"]["action_benefit_rate"] == 1.0
    assert result["attention"]["same_action_mass_mean"] == 0.25
    assert "high_motion" in result["motion_strata"]
    assert "traffic_incremental_action_map" in result["overall"]
