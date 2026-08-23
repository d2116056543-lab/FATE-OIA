import torch

from fate_oia.engine.evaluate_tida_oia import geometric_temporal_effectiveness_metrics


def test_geometric_effectiveness_reports_subsets_anticipation_and_transport():
    count = 12
    action_target = torch.randint(0, 2, (count, 4)).float()
    reason_target = torch.randint(0, 2, (count, 21)).float()
    image_action = (2 * action_target - 1) * 0.3
    image_reason = (2 * reason_target - 1) * 0.2
    action_delta = (2 * action_target - 1) * 0.1
    reason_delta = (2 * reason_target - 1) * 0.08
    rows = {
        "image_action": image_action, "semantic_action": image_action,
        "geometric_action": image_action + action_delta, "video_action": image_action + action_delta,
        "image_reason": image_reason, "semantic_reason": image_reason,
        "geometric_reason": image_reason + reason_delta, "video_reason": image_reason + reason_delta,
        "prefix_action": torch.stack([image_action + action_delta * scale for scale in (.25, .5, .75, 1)], 1),
        "prefix_reason": torch.stack([image_reason + reason_delta * scale for scale in (.25, .5, .75, 1)], 1),
        "action_target": action_target, "reason_target": reason_target,
        "geometric_motion_energy": torch.arange(count, dtype=torch.float32)[:, None].expand(-1, 4),
        "geometric_action_delta": action_delta, "geometric_reason_delta": reason_delta,
    }
    metrics = geometric_temporal_effectiveness_metrics(rows)
    assert metrics["subsets"]["high_motion"]["count"] > 0
    assert len(metrics["anticipation_curve"]) == 4
    assert metrics["target_transport"]["action_signed_margin_mean"] > 0
    assert metrics["target_transport"]["reason_signed_margin_mean"] > 0
