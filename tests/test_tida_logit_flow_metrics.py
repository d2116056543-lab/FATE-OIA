import torch

from fate_oia.engine.evaluate_tida_oia import logit_flow_effectiveness_metrics


def test_logit_flow_effectiveness_reports_target_transport_and_branch_ablation():
    action_target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    reason_target = action_target.clone()
    image_action = torch.zeros_like(action_target)
    image_reason = torch.zeros_like(reason_target)
    action_delta = (2.0 * action_target - 1.0) * 0.2
    action_delta[-1] = -action_delta[-1]
    reason_delta = (2.0 * reason_target - 1.0) * 0.1
    rows = {
        "image_action": image_action,
        "image_reason": image_reason,
        "video_action": image_action + action_delta,
        "video_reason": image_reason + reason_delta,
        "logit_flow_action_candidate_delta": action_delta,
        "logit_flow_reason_candidate_delta": reason_delta,
        "logit_flow_action_utility_probability": torch.tensor(
            [[0.8, 0.8], [0.8, 0.8], [0.8, 0.8], [0.2, 0.2]]
        ),
        "logit_flow_reason_utility_probability": torch.full_like(reason_delta, 0.7),
        "reason_contradiction_score": 1.0 - reason_target,
        "action_target": action_target,
        "reason_target": reason_target,
    }

    metrics = logit_flow_effectiveness_metrics(rows)

    assert metrics["action_transport"]["signed_margin_mean"] > 0
    assert metrics["reason_transport"]["observed_positive_margin_mean"] > 0
    assert metrics["branches"]["logit_flow_candidate"]["Act_mAP"] >= metrics["branches"]["image"]["Act_mAP"]
    assert metrics["utility"]["action_helpfulness_auc"] is not None


def test_reason_helpfulness_auc_excludes_pu_unknowns_and_uses_certified_direction():
    rows = {
        "image_action": torch.zeros(2, 1),
        "image_reason": torch.zeros(2, 3),
        "video_action": torch.zeros(2, 1),
        "video_reason": torch.zeros(2, 3),
        "logit_flow_action_candidate_delta": torch.zeros(2, 1),
        "logit_flow_reason_candidate_delta": torch.tensor(
            [[0.2, -0.2, 0.4], [0.1, 0.1, -0.4]]
        ),
        "logit_flow_action_utility_probability": torch.zeros(2, 1),
        "logit_flow_reason_utility_probability": torch.tensor(
            [[0.9, 0.8, 1.0], [0.7, 0.1, 0.0]]
        ),
        "action_target": torch.zeros(2, 1),
        "reason_target": torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "reason_contradiction_score": torch.tensor(
            [[0.0, 1.0, 0.2], [0.0, 1.0, 0.2]]
        ),
    }

    metrics = logit_flow_effectiveness_metrics(rows)

    assert abs(metrics["utility"]["reason_certified_coverage"] - 4 / 6) < 1e-6
    assert metrics["utility"]["reason_helpfulness_auc"] == 1.0
