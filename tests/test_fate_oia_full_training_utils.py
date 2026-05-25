from __future__ import annotations

import torch
from argparse import Namespace

from fate_oia.engine.train_fate_oia import (
    compress_tokens,
    compute_grounding_loss,
    counterfactual_deletion_loss,
    load_reason_grounding_rules,
    reason_to_action_consistency_loss,
    recover_label_attention,
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


def test_label_conditioned_grounding_uses_positive_reason(tmp_path):
    label_json = tmp_path / "sample.json"
    label_json.write_text(
        '{"frames":[{"objects":[{"category":"person","box2d":{"x1":0,"y1":0,"x2":8,"y2":8}}]}]}',
        encoding="utf-8",
    )
    rules_yaml = tmp_path / "rules.yaml"
    rules_yaml.write_text('reason_to_bdd100k_categories:\\n  6: ["person"]\\n', encoding="utf-8")
    rules = load_reason_grounding_rules(str(rules_yaml), reason_dim=21)
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
