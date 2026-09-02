import torch

from fate_oia.engine.evaluate_tida_oia import target_token_flow_effectiveness_metrics


def test_target_token_metrics_separate_candidate_deploy_and_controls():
    action_target = torch.zeros(2, 4)
    action_target[0, 0] = 1.0
    action_target[1, 1] = 1.0
    reason_target = torch.zeros(2, 21)
    reason_target[0, 0] = 1.0
    reason_target[1, 1] = 1.0
    image_action = torch.zeros(2, 4)
    image_reason = torch.zeros(2, 21)
    action_delta = torch.zeros(2, 4)
    action_delta[0, 0], action_delta[1, 1] = 0.4, 0.5
    action_delta[1, 0], action_delta[0, 1] = -0.2, -0.3
    reason_delta = torch.zeros(2, 21)
    reason_delta[0, 0], reason_delta[1, 1] = 0.3, 0.4
    rows = {
        "image_action": image_action,
        "image_reason": image_reason,
        "video_action": image_action + action_delta,
        "video_reason": image_reason + reason_delta,
        "target_token_action_candidate": image_action + action_delta,
        "target_token_action_deploy": image_action + action_delta,
        "target_token_reason_candidate": image_reason + reason_delta,
        "target_token_reason_deploy": image_reason + reason_delta,
        "target_token_action_candidate_delta": action_delta,
        "target_token_action_deploy_delta": action_delta,
        "target_token_reason_candidate_delta": reason_delta,
        "target_token_reason_deploy_delta": reason_delta,
        "target_token_action_deploy_gate": torch.ones(2, 4),
        "target_token_reason_deploy_gate": (reason_delta != 0).float(),
        "target_token_action_utility_probability": torch.full((2, 4), 0.8),
        "target_token_reason_utility_probability": torch.full((2, 21), 0.7),
        "target_token_action_ordered_prediction_error": torch.full((2, 4), 0.2),
        "target_token_action_reversed_prediction_error": torch.full((2, 4), 0.5),
        "target_token_action_repeated_prediction_error": torch.full((2, 4), 0.4),
        "target_token_action_shuffled_prediction_error": torch.full((2, 4), 0.45),
        "target_token_reason_ordered_prediction_error": torch.full((2, 21), 0.3),
        "target_token_reason_reversed_prediction_error": torch.full((2, 21), 0.6),
        "target_token_reason_repeated_prediction_error": torch.full((2, 21), 0.5),
        "target_token_reason_shuffled_prediction_error": torch.full((2, 21), 0.55),
        "action_target": action_target,
        "reason_target": reason_target,
    }

    result = target_token_flow_effectiveness_metrics(rows, torch.full((25,), 0.55))

    assert result["available"] is True
    assert result["action"]["candidate_signed_margin_mean"] > 0
    assert result["action"]["deploy_gate_nonzero_rate"] == 1.0
    assert result["reason"]["observed_positive_no_harm_rate"] == 1.0
    assert result["action"]["ordered_vs_reversed_advantage_mean"] > 0
    assert result["action"]["ordered_vs_repeated_win_rate"] == 1.0
    assert result["action"]["ordered_vs_shuffled_win_rate"] == 1.0
    assert result["action"]["utility_helpfulness_auc"] is not None
    assert result["decision_flips"]["action"]["fn_to_tp"] == [1, 1, 0, 0]
    assert set(result["metrics"]) == {"image", "candidate", "deploy"}


def test_target_token_metrics_are_unavailable_without_target_outputs():
    assert target_token_flow_effectiveness_metrics({}, 0.5) == {
        "available": False,
        "reason": "target_token_outputs_missing",
    }
