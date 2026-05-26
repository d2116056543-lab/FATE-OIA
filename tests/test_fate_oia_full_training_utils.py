from __future__ import annotations

import torch
from argparse import Namespace

from fate_oia.engine.train_fate_oia import (
    apply_config_defaults,
    action_branch_losses,
    build_scheduler,
    compress_tokens,
    compute_counterfactual_audit,
    compute_grounding_loss,
    counterfactual_deletion_loss,
    current_lr,
    load_config_defaults,
    load_reason_grounding_rules,
    reason_to_action_consistency_loss,
    recover_label_attention,
    scheduled_keep_ratio,
    step_scheduler,
    write_epoch_artifacts,
)
from fate_oia.losses.task_balance import UncertaintyTaskBalancer
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.utils.lr_scaling import compute_lr_scaling, effective_batch_size, scale_lr_linear


def test_compress_tokens_preserves_cls_and_provenance_rows():
    tokens = torch.randn(2, 17, 8)
    reduced, provenance, stats = compress_tokens(tokens, keep_ratio=0.5, num_summary_tokens=2, min_tokens=4)
    assert reduced.shape[0] == 2
    assert reduced.shape[-1] == 8
    assert provenance is not None
    assert provenance.shape[:2] == (2, 17)
    assert torch.allclose(provenance.sum(-1), torch.ones(2, 17), atol=1e-6)
    assert torch.allclose(reduced[:, 0], tokens[:, 0])
    assert stats["enabled"] is True


def test_compression_schedule_is_disabled_before_start_epoch():
    args = Namespace(
        token_compression="keep_merge",
        compression_start_epoch=8,
        compression_warmup_epochs=6,
        compression_keep_ratio_start=0.85,
        compression_keep_ratio_final=0.65,
    )
    assert scheduled_keep_ratio(args, 0) == 1.0
    assert scheduled_keep_ratio(args, 8) == 0.85
    assert scheduled_keep_ratio(args, 14) == 0.65


def test_lr_scaling_effective_batch_and_cap():
    assert effective_batch_size(per_gpu_batch_size=2, gradient_accumulation_steps=16, num_gpus=1) == 32
    assert scale_lr_linear(3e-4, effective_batch=32, reference_effective_batch=32) == 3e-4
    result = compute_lr_scaling(
        per_gpu_batch_size=4,
        gradient_accumulation_steps=8,
        reference_effective_batch=32,
        base_lr_at_reference_batch=3e-4,
        max_lr=5e-4,
    )
    assert result.effective_batch_size == 32
    assert result.lr_actual == 3e-4
    capped = compute_lr_scaling(
        per_gpu_batch_size=4,
        gradient_accumulation_steps=16,
        reference_effective_batch=32,
        base_lr_at_reference_batch=3e-4,
        max_lr=5e-4,
    )
    assert capped.effective_batch_size == 64
    assert capped.lr_actual == 5e-4


def test_config_defaults_apply_when_cli_does_not_override(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("data:\n  image_height: 360\ntraining:\n  batch_size: 2\n  auto_scale_lr: true\n", encoding="utf-8")
    args = Namespace(config=str(cfg), image_height=224, batch_size=8, auto_scale_lr=False)
    monkeypatch.setattr("sys.argv", ["train_fate_oia.py", "--config", str(cfg)])
    apply_config_defaults(args, load_config_defaults(str(cfg)))
    assert args.image_height == 360
    assert args.batch_size == 2
    assert args.auto_scale_lr is True


def test_config_defaults_respect_boolean_optional_negative_flag(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("training:\n  auto_scale_lr: true\n", encoding="utf-8")
    args = Namespace(config=str(cfg), auto_scale_lr=False)
    monkeypatch.setattr("sys.argv", ["train_fate_oia.py", "--config", str(cfg), "--no-auto_scale_lr"])
    apply_config_defaults(args, load_config_defaults(str(cfg)))
    assert args.auto_scale_lr is False


def test_fate_oia_feature_model_outputs_attention_and_r2a():
    model = FATEOIAFeatureModel(dim=16, action_dim=4, reason_dim=21, use_label_query=True)
    out = model(torch.randn(3, 10, 16))
    assert out["action_logits"].shape == (3, 4)
    assert out["reason_logits"].shape == (3, 21)
    assert out["reason_to_action_logits"].shape == (3, 4)
    assert out["attention"].shape[:3] == (3, 4, 25)


def test_label_correlation_block_changes_label_tokens_and_shapes():
    torch.manual_seed(0)
    model = FATEOIAFeatureModel(
        dim=16,
        action_dim=4,
        reason_dim=21,
        use_label_query=True,
        label_correlation="self_attn",
        label_correlation_layers=1,
        label_correlation_heads=4,
    )
    out = model(torch.randn(2, 12, 16))
    assert out["action_logits"].shape == (2, 4)
    assert out["reason_logits"].shape == (2, 21)
    assert out["label_tokens"].shape == (2, 25, 16)


def test_recover_label_attention_from_compressed_tokens():
    tokens = torch.randn(1, 9, 4)
    reduced, provenance, _ = compress_tokens(tokens, keep_ratio=0.5, num_summary_tokens=1, min_tokens=2)
    attention = torch.softmax(torch.randn(1, 2, 25, reduced.shape[1]), dim=-1)
    recovered = recover_label_attention(attention, provenance, original_tokens=tokens.shape[1])
    assert recovered.shape == (1, 25, 9)


def test_reason_to_action_consistency_loss_is_scalar():
    loss = reason_to_action_consistency_loss(torch.randn(5, 4), torch.randn(5, 4))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_action_branch_loss_excludes_fused_aux_by_default():
    out = {
        "action_visual_logits": torch.tensor([[0.2, -0.1, 0.3, -0.4]], requires_grad=True),
        "action_reason_logits": torch.tensor([[-0.2, 0.1, -0.3, 0.4]], requires_grad=True),
        "action_fused_logits": torch.tensor([[2.0, -2.0, 2.0, -2.0]], requires_grad=True),
        "action_logits": torch.tensor([[2.0, -2.0, 2.0, -2.0]], requires_grad=True),
    }
    target = torch.tensor([[1.0, 0.0, 0.0, 1.0]])
    losses = action_branch_losses(
        out,
        target,
        loss_action_visual=0.05,
        loss_r2a_gt=0.10,
        loss_action_agree=0.01,
        include_fused_branch_loss=False,
    )
    expected = (
        losses["action_visual_loss"] * 0.05
        + losses["action_reason_loss"] * 0.10
        + losses["action_agree_loss"] * 0.01
    )
    assert torch.allclose(losses["action_branch_total"], expected)
    assert losses["action_fused_loss_main_only"].detach() > 0


def test_counterfactual_deletion_loss_backpropagates_to_model():
    torch.manual_seed(0)
    model = FATEOIAFeatureModel(dim=8, action_dim=4, reason_dim=21, use_label_query=True)
    tokens = torch.randn(2, 7, 8)
    labels = torch.randint(0, 2, (2, 25)).float()
    base_loss = torch.tensor(0.0)
    loss = counterfactual_deletion_loss(model, tokens, labels, base_loss, action_dim=4, topk_ratio=0.5, margin=10.0)
    loss.backward()
    grad_norm = sum(float(p.grad.detach().abs().sum().item()) for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0


def test_counterfactual_deletion_loss_isolated_gradient_with_mean_fill():
    torch.manual_seed(1)
    model = FATEOIAFeatureModel(dim=8, action_dim=4, reason_dim=21, use_label_query=True)
    tokens = torch.randn(2, 9, 8)
    labels = torch.randint(0, 2, (2, 25)).float()
    loss = counterfactual_deletion_loss(
        model,
        tokens,
        labels,
        base_loss=torch.tensor(0.0),
        action_dim=4,
        topk_ratio=0.4,
        margin=10.0,
        mask_fill="mean",
    )
    loss.backward()
    grad_norm = sum(float(p.grad.detach().abs().sum().item()) for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0


def test_counterfactual_audit_reports_drop_fields_without_loss():
    torch.manual_seed(2)
    model = FATEOIAFeatureModel(dim=8, action_dim=4, reason_dim=21, use_label_query=True)
    tokens = torch.randn(2, 9, 8)
    labels = torch.randint(0, 2, (2, 25)).float()
    stats = compute_counterfactual_audit(model, tokens, labels, action_dim=4, topk_ratio=0.4, mask_fill="mean")
    assert stats["cf_valid_count"] == 2
    for key in ["cf_action_drop_mean", "cf_reason_drop_mean", "cf_random_action_drop_mean", "cf_random_reason_drop_mean", "cf_base_prob", "cf_masked_prob"]:
        assert key in stats


def test_uncertainty_task_balancer_has_gradients():
    balancer = UncertaintyTaskBalancer(("action", "reason", "r2a", "grounding"))
    total, components = balancer({
        "action": torch.tensor(1.0, requires_grad=True),
        "reason": torch.tensor(2.0, requires_grad=True),
        "r2a": torch.tensor(0.5, requires_grad=True),
        "grounding": torch.tensor(0.1, requires_grad=True),
    })
    total.backward()
    assert balancer.log_vars["action"].grad is not None
    assert "task_balance_reason_weighted" in components


def test_cosine_scheduler_steps_and_can_restore_state():
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    args = Namespace(scheduler="cosine", epochs=4, start_epoch=1, min_lr=1e-5, warmup_epochs=0)
    sched = build_scheduler(args, opt)
    assert sched is not None
    before = current_lr(opt)
    step_scheduler(args, sched, val_score=0.1, test_score=0.2, row={"val_metrics": {}, "test_metrics": {}})
    after = current_lr(opt)
    assert after < before
    state = sched.state_dict()
    opt2 = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched2 = build_scheduler(args, opt2)
    sched2.load_state_dict(state)
    assert sched2.state_dict()["last_epoch"] == state["last_epoch"]


def test_write_epoch_artifacts_creates_diagnostic_files(tmp_path):
    logits = torch.randn(2, 25)
    labels = torch.randint(0, 2, (2, 25)).float()
    stats = {
        "metrics": {"Act_mF1": 0.0, "Exp_mF1": 0.0},
        "branch_metrics": {"action_fused": {"Act_mF1": 0.0}},
        "logits": logits,
        "visual_logits": logits[:, :4],
        "reason_action_logits": logits[:, :4],
        "fused_logits": logits[:, :4],
        "labels": labels,
        "file_names": ["a.jpg", "b.jpg"],
        "token_stats": [{"original_tokens": 10, "reduced_tokens": 10, "compression_active": False}],
        "loss_components": [{"step": 0, "main_loss": 1.0, "action_branch_total": 0.1}],
        "grounding_stats": [{"grounding_valid_count": 0}],
        "counterfactual_stats": [{"cf_loss": 0.0, "cf_valid_count": 0}],
    }
    write_epoch_artifacts(tmp_path, 0, stats, stats, {"command": "pytest", "is_smoke": True})
    epoch_dir = tmp_path / "epoch_000"
    for name in [
        "metrics_summary.json",
        "loss_components.jsonl",
        "branch_metrics.json",
        "logits_visual_action.pt",
        "logits_reason_action.pt",
        "logits_fused_action.pt",
        "logits_reason.pt",
        "labels_action.pt",
        "labels_reason.pt",
        "file_names.json",
        "token_stats.jsonl",
        "grounding_stats.jsonl",
        "counterfactual_stats.jsonl",
        "fusion_stats.json",
        "failure_cases.jsonl",
    ]:
        assert (epoch_dir / name).exists(), name


def test_label_conditioned_grounding_uses_positive_reason(tmp_path):
    label_json = tmp_path / "sample.json"
    label_json.write_text(
        '{"frames":[{"objects":[{"category":"person","box2d":{"x1":0,"y1":0,"x2":8,"y2":8}}]}]}',
        encoding="utf-8",
    )
    rules_yaml = tmp_path / "rules.yaml"
    rules_yaml.write_text('reason_to_bdd100k_categories:\n  6: ["person"]\n', encoding="utf-8")
    rules = load_reason_grounding_rules(str(rules_yaml), reason_dim=21)
    assert rules == {6: {"person"}}
    args = Namespace(
        action_dim=4,
        reason_dim=21,
        image_height=16,
        image_width=16,
        patch_size=8,
        grounding_image_width=16,
        grounding_image_height=16,
        grounding_categories="person",
        grounding_mode="label",
        reason_grounding_rules_map=rules,
    )
    attention = torch.zeros(1, 25, 5)
    attention[:, :, 1:] = 0.25
    batch = {
        "file_name": ["sample.jpg"],
        "reason": torch.tensor([[0, 0, 0, 0, 0, 0, 1] + [0] * 14], dtype=torch.float32),
    }
    cache = {"sample.jpg": {"file_name": "sample.jpg", "label_json": str(label_json)}}
    loss, stats = compute_grounding_loss(attention, batch, cache, args, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert stats["reason_6_count"] == 1.0


def test_label_conditioned_grounding_uses_lane_and_drivable_targets(tmp_path):
    label_json = tmp_path / "sample.json"
    label_json.write_text(
        '{"frames":[{"objects":['
        '{"category":"lane/single white","poly2d":[{"vertices":[[0,8],[16,8],[16,12],[0,12]],"closed":true}]},'
        '{"category":"area/drivable","poly2d":[{"vertices":[[0,12],[16,12],[16,16],[0,16]],"closed":true}]}'
        ']}]}',
        encoding="utf-8",
    )
    drive_map = tmp_path / "drive.png"
    from PIL import Image

    image = Image.new("L", (16, 16), 0)
    for x in range(16):
        for y in range(12, 16):
            image.putpixel((x, y), 1)
    image.save(drive_map)
    rules_yaml = tmp_path / "rules.yaml"
    rules_yaml.write_text(
        'reason_to_bdd100k_categories:\n'
        '  2: ["area/drivable"]\n'
        '  11: ["lane/single white"]\n',
        encoding="utf-8",
    )
    rules = load_reason_grounding_rules(str(rules_yaml), reason_dim=21)
    args = Namespace(
        action_dim=4,
        reason_dim=21,
        image_height=16,
        image_width=16,
        patch_size=8,
        grounding_image_width=16,
        grounding_image_height=16,
        grounding_categories="",
        grounding_mode="label",
        reason_grounding_rules_map=rules,
    )
    attention = torch.zeros(1, 25, 5)
    attention[:, :, 1:] = 0.25
    reason = torch.zeros(1, 21)
    reason[0, 2] = 1.0
    reason[0, 11] = 1.0
    batch = {"file_name": ["sample.jpg"], "reason": reason}
    cache = {"sample.jpg": {"file_name": "sample.jpg", "label_json": str(label_json), "drivable_map": str(drive_map)}}
    loss, stats = compute_grounding_loss(attention, batch, cache, args, torch.device("cpu"))
    assert torch.isfinite(loss)
    assert stats["reason_2_count"] == 1.0
    assert stats["reason_2_drivable_count"] == 1.0
    assert stats["reason_11_count"] == 1.0
    assert stats["reason_11_lane_count"] == 1.0
    assert stats["grounding_drivable_count"] >= 1.0
    assert stats["grounding_lane_count"] >= 1.0
