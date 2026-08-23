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
    assert len(result["dynamic_benefit_curve"]) == 5
    assert result["corrective_flip_counts_by_action"]["fp_to_tn"] == [1, 1, 1, 1]
    assert result["corrective_flip_counts_by_action"]["tp_to_fn"] == [0, 0, 0, 0]


def test_traffic_effectiveness_reduces_common_motion_over_batch_and_time():
    target = torch.zeros(2, 4)
    common = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    rows = {
        "image_action": torch.zeros(2, 4),
        "semantic_action": torch.zeros(2, 4),
        "traffic_action": torch.zeros(2, 4),
        "video_action": torch.zeros(2, 4),
        "image_reason": torch.zeros(2, 21),
        "semantic_reason": torch.zeros(2, 21),
        "video_reason": torch.zeros(2, 21),
        "action_target": target,
        "reason_target": torch.zeros(2, 21),
        "traffic_action_delta": torch.zeros(2, 4),
        "traffic_motion_energy": torch.ones(2, 3),
        "traffic_action_attention": torch.full((2, 4, 12), 1.0 / 12),
        "traffic_same_action_mass": torch.full((2, 4), 0.25),
        "traffic_patch_displacement": torch.zeros(2, 3, 4, 2),
        "traffic_patch_common_displacement": common,
        "traffic_patch_exclusive_displacement": torch.zeros(2, 3, 4, 2),
        "traffic_patch_match_confidence": torch.ones(2, 3, 4),
        "traffic_patch_motion_energy": torch.ones(2, 3, 4),
    }
    result = traffic_action_effectiveness_metrics(rows)
    assert result["attention"]["patch_common_displacement_xy_mean"] == common.mean((0, 1)).tolist()
