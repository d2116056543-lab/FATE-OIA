import pytest
import torch

from fate_oia.engine.collect_vetra_tta_outputs import checkpoint_inference_scales
from fate_oia.engine.train_aie_oia import (
    checkpoint_selection_criteria,
    compatible_checkpoint_state_dict,
    deployment_metrics_from_logits,
    load_config,
    schedule_values,
)


def test_checkpoint_selection_tracks_genuine_fixed_metrics_separately_from_deploy():
    selected = {
        "final": {"joint": 0.54, "Act_mF1": 0.72, "Exp_mF1": 0.36, "Act_mAP": 0.79, "Exp_mAP": 0.38},
        "deploy": {"joint": 0.57, "Act_mF1": 0.73, "Exp_mF1": 0.41},
    }

    criteria = checkpoint_selection_criteria(selected)

    assert criteria["fixed_joint"] == 0.54
    assert criteria["fixed_action_mF1"] == 0.72
    assert criteria["fixed_reason_mF1"] == 0.36
    assert criteria["deploy_joint"] == 0.57


def test_train_audit_deployment_metrics_apply_train_calib_thresholds():
    action_logits = torch.tensor([[2.0, -2.0, -2.0, -2.0], [-2.0, 2.0, -2.0, -2.0]])
    reason_logits = torch.full((2, 21), -2.0)
    action_target = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    reason_target = torch.zeros(2, 21)
    thresholds = {"threshold_prob": torch.full((25,), 0.5)}

    metrics = deployment_metrics_from_logits(
        action_logits, reason_logits, action_target, reason_target, thresholds
    )

    assert metrics["final"]["Act_mF1"] == pytest.approx(0.5)
    assert metrics["deploy"]["Act_mF1"] == pytest.approx(0.5)


def test_clean_single_run_config_never_selects_or_trains_on_held_out_splits():
    cfg = load_config("configs/fate_oia_train_360x640_vetra_clean_single_run.yaml")

    assert cfg["experiment"]["best_selection_split"] == "train_audit"
    assert cfg["experiment"]["internal_test_selected"] is False
    assert cfg["calibration"]["exclude_from_training"] is True
    assert cfg["data"]["train_on_all_train"] is False


def test_tta_collection_uses_checkpoint_scales_not_schedule_start_values():
    cfg = load_config("configs/fate_oia_train_360x640_vetra_clean_single_run.yaml")
    checkpoint = {"inference_scales": {"action": 0.85, "reason": 0.55}}

    assert checkpoint_inference_scales(checkpoint, cfg) == (0.85, 0.55)


def test_checkpoint_loader_drops_only_known_nonpersistent_grammar_buffers():
    model = torch.nn.Linear(2, 1)
    state = model.state_dict()
    state["foundation.predicate_reason.positive_mask"] = torch.ones(1)

    compatible = compatible_checkpoint_state_dict(model, state)

    assert set(compatible) == set(model.state_dict())

    state["unexpected.learned.weight"] = torch.ones(1)
    with pytest.raises(RuntimeError, match="unexpected learned state"):
        compatible_checkpoint_state_dict(model, state)


def test_proper_recovery_preserves_strong_checkpoint_branch_scales_from_first_step():
    cfg = load_config("configs/fate_oia_train_360x640_vetra_trainable_073_040_v2_proper_recovery.yaml")

    schedule = schedule_values(0, 100, cfg)

    assert schedule["action"] == 1.0
    assert schedule["reason"] == 0.6


def test_reason_recovery_freezes_every_action_owner_and_keeps_reason_proper_loss():
    cfg = load_config("configs/fate_oia_train_360x640_vetra_trainable_073_040_v2_reason_recovery.yaml")

    assert cfg["training"]["trainable_owners"] == ["reason_private"]
    assert "final_action_calibration" not in cfg["loss_weights"]
    assert cfg["loss_weights"]["final_reason_calibration"] > 0
    assert schedule_values(0, 100, cfg)["action"] == 1.0
    assert schedule_values(0, 100, cfg)["reason"] == 0.6
