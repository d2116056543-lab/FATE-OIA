import inspect
from pathlib import Path

import yaml
import torch

import fate_oia.engine.supervise_cast_oia_foreground as supervisor
import fate_oia.engine.train_cast_oia as train_cast


def test_config_protocol_values():
    cfg = yaml.safe_load(Path("configs/fate_oia_train_360x640_cast_oia_v1.yaml").read_text(encoding="utf-8"))
    assert cfg["training"]["batch_size"] == 5
    assert cfg["training"]["gradient_accumulation_steps"] == 6
    assert cfg["training"]["reference_effective_batch"] == 32
    assert cfg["training"]["warmup_epochs"] == 3
    assert cfg["model"]["token_compression"] == "none"
    assert cfg["model"]["feature_cache_enabled"] is False
    assert cfg["data"]["eval_splits"] == "test"
    assert cfg["training"]["best_selection_split"] == "test"


def test_foreground_supervisor_forbidden_patterns_absent():
    src = inspect.getsource(supervisor)
    forbidden = ["Start-Process", "Start-Job", "nohup", "Register-ScheduledTask"]
    assert not any(x in src for x in forbidden)
    assert "FALLBACK_LADDER" in src
    assert "(5, 6)" in src and "(2, 16)" in src


def test_train_protocol_uses_direct_image_test_only_no_cache():
    src = inspect.getsource(train_cast)
    assert "BDDOIAMultiTaskDataset" in src
    assert "load_image=True" in src
    assert "eval_splits" in src
    assert "feature_cache_enabled" in src
    assert "token_compression" in src
    assert "checkpoint_best_test.pth" in src


def test_supervisor_requires_review_pass_bound_to_current_head():
    src = inspect.getsource(supervisor)
    assert "cast_oia_v1_preflight_postcommit" in src
    assert "git rev-parse HEAD" in src
    assert "REVIEW_PASS_CAST_OIA_V1.txt" in src


def test_train_supports_checkpoint_resume_without_resetting_best_scores():
    src = inspect.getsource(train_cast)
    assert "--resume_checkpoint" in src
    assert "_load_resume_state" in src
    assert "_load_best_scores_from_metrics" in src
    assert "start_epoch = resume_epoch + 1" in src
    assert "range(start_epoch" in src
    assert "checkpoint_latest.pth" in src


def test_supervisor_and_launcher_forward_resume_checkpoint():
    supervisor_src = inspect.getsource(supervisor)
    launcher_src = Path("scripts/FATE_OIA_cast_oia_v1_foreground.ps1").read_text(encoding="utf-8")
    assert "--resume_checkpoint" in supervisor_src
    assert "args.resume_checkpoint" in supervisor_src
    assert "[string]$ResumeCheckpoint" in launcher_src
    assert "--resume_checkpoint" in launcher_src


def test_resume_filters_only_legacy_frozen_dino_vproj_keys():
    state = {
        "head.weight": torch.ones(1),
        "dino.backbone.blocks.0.attn.vproj.weight": torch.ones(1),
        "dino.backbone.blocks.0.attn.vproj.bias": torch.zeros(1),
    }
    filtered, removed = train_cast._strip_legacy_dino_vproj_keys(state)
    assert "head.weight" in filtered
    assert "dino.backbone.blocks.0.attn.vproj.weight" not in filtered
    assert "dino.backbone.blocks.0.attn.vproj.bias" not in filtered
    assert removed == [
        "dino.backbone.blocks.0.attn.vproj.weight",
        "dino.backbone.blocks.0.attn.vproj.bias",
    ]



def test_reason_warmup_weights_are_epoch_gated():
    early = train_cast.loss_weights_for_epoch(0)
    mid = train_cast.loss_weights_for_epoch(4)
    light = train_cast.loss_weights_for_epoch(11)
    late = train_cast.loss_weights_for_epoch(21)
    assert early["reason"] == 1.20
    assert early["action_set"] == 0.45
    assert early["drop_add"] == 0.15
    assert "reason_sigmoid_f1" in early
    assert early["reason_sigmoid_f1"] >= 0.08
    assert early["reason_positive_boost"] == 3.0
    assert early["reason_negative_scale"] == 0.5
    assert mid["reason"] == 1.05
    assert mid["action_set"] == 0.50
    assert mid["drop_add"] == 0.20
    assert mid["reason_positive_boost"] == 2.0
    assert mid["reason_negative_scale"] == 0.65
    assert light["reason"] == 0.95
    assert light["reason_positive_boost"] == 1.5
    assert light["reason_negative_scale"] == 0.8
    assert late["reason"] == 0.85
    assert late["action_set"] == 0.60
    assert late["drop_add"] == 0.25
    assert late["reason_positive_boost"] == 1.0
    assert late["reason_negative_scale"] == 1.0


def test_evaluate_outputs_reason_threshold_and_positive_rate_diagnostics():
    outputs = {
        "action_logits": torch.randn(6, 4),
        "reason_logits": torch.randn(6, 21),
        "action_set_probs": torch.softmax(torch.randn(6, 16), dim=-1),
    }
    labels_action = torch.randint(0, 2, (6, 4)).float()
    labels_reason = torch.randint(0, 2, (6, 21)).float()
    metrics = train_cast.evaluate_cast_outputs(outputs, labels_action, labels_reason)
    for key in [
        "Exp_mF1_fixed_0.5",
        "Exp_mF1_global_threshold_best",
        "Exp_mF1_per_label_threshold_best",
        "Exp_global_threshold_best",
        "reason_gt_positive_rate",
        "reason_pred_positive_rate@0.5",
        "reason_pred_positive_rate@0.3",
        "reason_pred_positive_rate@0.2",
    ]:
        assert key in metrics
