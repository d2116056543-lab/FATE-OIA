import torch

from fate_oia.utils.tida_relational_traffic_metrics import relational_traffic_metrics


def test_relational_metrics_measure_transport_not_just_ablation():
    action_target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    reason_target = action_target.clone()
    base_action = torch.zeros(4, 2)
    base_reason = torch.zeros(4, 2)
    sign = 2.0 * action_target - 1.0
    action_delta = 0.4 * sign
    reason_delta = 0.3 * sign
    rows = {
        "pre_relational_action": base_action,
        "pre_relational_reason": base_reason,
        "video_action": base_action + action_delta,
        "video_reason": base_reason + reason_delta,
        "action_target": action_target,
        "reason_target": reason_target,
        "relational_action_delta": action_delta,
        "relational_reason_delta": reason_delta,
        "relational_action_selected_deleted_delta": torch.zeros_like(action_delta),
        "relational_action_random_deleted_delta": 0.75 * action_delta,
        "relational_reason_selected_deleted_delta": torch.zeros_like(reason_delta),
        "relational_reason_random_deleted_delta": 0.75 * reason_delta,
        "relational_action_support": torch.ones_like(action_delta),
        "relational_reason_support": torch.ones_like(reason_delta),
        "relational_action_attention": torch.full((4, 2, 3), 1.0 / 3),
        "relational_reason_attention": torch.full((4, 2, 3), 1.0 / 3),
        "relational_interaction_risk": torch.rand(4, 3, 3),
    }
    metrics = relational_traffic_metrics(rows, bootstrap_samples=100)
    assert metrics["action"]["conditional_nll_improvement"] > 0
    assert metrics["action"]["conditional_information_gain_bits"] > 0
    assert metrics["action"]["relative_brier_reduction"] > 0
    assert metrics["reason"]["conditional_nll_improvement"] > 0
    assert metrics["action"]["selected_minus_random_deletion_gap"] > 0
    assert metrics["reason"]["selected_minus_random_deletion_gap"] > 0
    assert metrics["action"]["signed_margin_benefit_rate"] == 1.0
    assert metrics["high_interaction_risk"]["available"]
    assert metrics["high_interaction_risk"]["action_information_gain_bits"] > 0
    assert metrics["high_interaction_risk"]["action_deletion_gap"] > 0
    assert metrics["action"]["route_necessity_precision"] == 1.0
    assert len(metrics["interaction_risk_quartiles"]) == 4


def test_relational_metrics_show_utility_concentrated_in_high_risk_clips():
    target = torch.ones(8, 1)
    risk = torch.arange(8, dtype=torch.float32) / 7.0
    delta = (0.01 + 0.08 * risk)[:, None]
    rows = {
        "pre_relational_action": torch.zeros_like(target),
        "pre_relational_reason": torch.zeros_like(target),
        "video_action": delta,
        "video_reason": delta,
        "action_target": target,
        "reason_target": target,
        "relational_action_delta": delta,
        "relational_reason_delta": delta,
        "relational_action_selected_deleted_delta": torch.zeros_like(delta),
        "relational_action_random_deleted_delta": 0.8 * delta,
        "relational_reason_selected_deleted_delta": torch.zeros_like(delta),
        "relational_reason_random_deleted_delta": 0.8 * delta,
        "relational_action_support": torch.ones_like(target),
        "relational_reason_support": torch.ones_like(target),
        "relational_action_attention": torch.ones(8, 1, 1),
        "relational_reason_attention": torch.ones(8, 1, 1),
        "relational_interaction_risk": risk[:, None, None],
    }
    metrics = relational_traffic_metrics(rows, bootstrap_samples=50)
    quartiles = metrics["interaction_risk_quartiles"]
    assert quartiles[-1]["action_information_gain_bits"] > quartiles[0]["action_information_gain_bits"]
    assert metrics["risk_utility_association"]["action_spearman"] > 0.9
    assert metrics["risk_utility_association"]["high_minus_low_action_information_gain_bits"] > 0
