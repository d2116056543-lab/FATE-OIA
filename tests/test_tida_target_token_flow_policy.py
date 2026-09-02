from types import SimpleNamespace

import torch

from fate_oia.engine.train_tida_oia import calibrate_target_token_flow_deployment
from fate_oia.models.tida_target_token_flow import TIDATargetTokenFlow


def test_target_token_policy_uses_train_calib_and_keeps_zero_fallback():
    rows = 20
    action_target = torch.zeros(rows, 4)
    reason_target = torch.zeros(rows, 21)
    action_target[::2, 0] = 1.0
    reason_target[::2, 0] = 1.0
    image_action = torch.where(action_target > 0, 2.0, -2.0)
    image_reason = torch.where(reason_target > 0, 2.0, -2.0)
    calib = {
        "action_target": action_target,
        "reason_target": reason_target,
        "image_action": image_action,
        "image_reason": image_reason,
        "video_action": image_action.clone(),
        "video_reason": image_reason.clone(),
        "target_token_action_candidate_delta": torch.zeros_like(action_target),
        "target_token_reason_candidate_delta": torch.zeros_like(reason_target),
        "target_token_action_utility_probability": torch.full_like(action_target, 0.5),
        "target_token_reason_utility_probability": torch.full_like(reason_target, 0.5),
    }
    model = SimpleNamespace(
        target_token_flow_enabled=True,
        target_token_action=TIDATargetTokenFlow(4, dim=4, hidden_dim=4),
        target_token_reason=TIDATargetTokenFlow(21, dim=4, hidden_dim=4),
    )
    deployment = {
        "locked_image_thresholds": [0.5] * 25,
        "locked_image_threshold_source": "train_calib_fixture",
        "target_token_action_policy_scales": [-1.0, 0.0, 1.0],
        "target_token_reason_policy_scales": [-1.0, 0.0, 1.0],
        "target_token_action_policy_cutoffs": [0.0, 0.5],
        "target_token_reason_policy_cutoffs": [0.0, 0.5],
        "target_token_policy_oof_folds": 2,
        "target_token_action_policy_min_oof_gain": 0.001,
        "target_token_reason_policy_min_oof_gain": 0.001,
    }

    result = calibrate_target_token_flow_deployment(model, calib, deployment)

    assert result["source"] == "train_calib_oof"
    assert result["test_labels_used"] is False
    assert not torch.count_nonzero(model.target_token_action.deployment_label_gate)
    assert not torch.count_nonzero(model.target_token_reason.deployment_label_gate)
    assert not torch.count_nonzero(model.target_token_action.deployment_scale)
    assert not torch.count_nonzero(model.target_token_reason.deployment_scale)
