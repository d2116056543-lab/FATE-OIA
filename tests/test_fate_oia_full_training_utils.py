from __future__ import annotations

import torch
from argparse import Namespace

from fate_oia.engine.train_fate_oia import (
    action_branch_losses,
    compress_tokens,
    compute_grounding_loss,
    counterfactual_deletion_loss,
    load_reason_grounding_rules,
    reason_to_action_consistency_loss,
    recover_label_attention,
    scheduled_keep_ratio,
    write_epoch_artifacts,
)
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel


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


def test_fate_oia_feature_model_outputs_attention_and_r2a():
    model = FATEOIAFeatureModel(dim=16, action_dim=4, reason_dim=21, use_label_query=True)
    out = model(torch.randn(3, 10, 16))
    assert out["action_logits"].shape == (3, 4)
    assert out["reason_logits"].shape == (3, 21)
    assert out["reason_to_action_logits"].shape == (3, 4)
    assert out["attention"].shape[:3] == (3, 4, 25)


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
