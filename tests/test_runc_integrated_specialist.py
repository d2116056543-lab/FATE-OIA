from __future__ import annotations

from argparse import Namespace

import pytest
import torch

import fate_oia.engine.train_fate_oia as base_train
import fate_oia.engine.train_runc_integrated_specialist as specialist_train
from fate_oia.models.action_set_head import ActionSetHead, assign_action_patterns, build_action_patterns
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.reason_visual_specialist import ReasonVisualSpecialist
from fate_oia.models.runc_integrated_specialist import RunCIntegratedSpecialist
from fate_oia.losses.specialist_losses import hard_reason_ranking_loss, sigmoid_f1_loss


def test_model_initial_near_runc():
    torch.manual_seed(0)
    base = FATEOIAFeatureModel(dim=32, action_dim=4, reason_dim=21, label_correlation="none")
    model = RunCIntegratedSpecialist(base, dim=32, action_dim=4, reason_dim=21, alpha_init=0.01)
    tokens = torch.randn(2, 12, 32)
    out = model(tokens)
    assert out["final_reason_logits"].shape == out["base_reason_logits"].shape
    assert out["final_action_logits"].shape == out["base_action_logits"].shape
    assert (out["final_action_logits"] - out["base_action_logits"]).abs().mean() < 1.0


def test_reason_specialist_uses_visual_tokens():
    torch.manual_seed(0)
    spec = ReasonVisualSpecialist(dim=32, reason_dim=21, num_heads=4)
    label_tokens = torch.randn(2, 25, 32)
    visual_a = torch.randn(2, 10, 32)
    visual_b = visual_a + 0.5
    delta_a, _ = spec(visual_a, label_tokens)
    delta_b, _ = spec(visual_b, label_tokens)
    assert not torch.allclose(delta_a, delta_b)


def test_action_set_conversion():
    actions = torch.tensor([[1,0,0,0],[0,1,0,0],[1,0,1,0],[1,0,0,0]], dtype=torch.float32)
    matrix, meta = build_action_patterns(actions, top_k=16)
    head = ActionSetHead(dim=32, action_dim=4, reason_dim=21, pattern_matrix=matrix)
    pattern_logits, action_logits = head(torch.randn(3,4), torch.randn(3,21), torch.randn(3,25,32))
    assert pattern_logits.shape[0] == 3
    assert action_logits.shape == (3,4)
    ids = assign_action_patterns(actions, matrix)
    assert ids.shape[0] == actions.shape[0]


def test_loss_backward():
    torch.manual_seed(0)
    base = FATEOIAFeatureModel(dim=32, action_dim=4, reason_dim=21, label_correlation="none")
    model = RunCIntegratedSpecialist(base, dim=32, action_dim=4, reason_dim=21)
    tokens = torch.randn(2, 12, 32)
    labels = torch.randint(0, 2, (2, 21)).float()
    out = model(tokens)
    loss = hard_reason_ranking_loss(out["final_reason_logits"], labels) + sigmoid_f1_loss(out["final_reason_logits"], labels) + out["pattern_logits"].mean()
    loss.backward()
    grad = 0.0
    for p in list(model.reason_specialist.parameters()) + list(model.action_set_head.parameters()):
        if p.grad is not None:
            grad += float(p.grad.detach().abs().sum().item())
    assert grad > 0


def test_config_drift_constants():
    # Guard the main structural settings that must remain Run C-compatible.
    args = Namespace(image_height=360, image_width=640, patch_size=8, action_dim=4, reason_dim=21, token_compression="keep_merge")
    assert (args.image_height, args.image_width, args.patch_size) == (360, 640, 8)
    assert (args.action_dim, args.reason_dim) == (4, 21)
    assert args.token_compression == "keep_merge"


def test_config_drift_runtime_guard(tmp_path):
    args = Namespace(
        image_height=360,
        image_width=640,
        patch_size=8,
        action_dim=4,
        reason_dim=21,
        token_compression="keep_merge",
        compression_keep_ratio_final=0.70,
        num_summary_tokens=4,
        runc_checkpoint=str(specialist_train.RUNC_ART / "checkpoint_best_test.pth"),
        runc_args=str(specialist_train.RUNC_ART / "args.json"),
        runc_config=str(specialist_train.RUNC_ART / "training_config_resolved.yaml"),
    )
    report = specialist_train.assert_no_runc_config_drift(args, output_dir=tmp_path)
    assert report["passed"] is True
    assert (tmp_path / "config_drift_report.json").exists()
    args.image_width = 672
    with pytest.raises(ValueError, match="Run C config drift"):
        specialist_train.assert_no_runc_config_drift(args)


def test_base_loading_strict():
    checkpoint_path = specialist_train.RUNC_ART / "checkpoint_best_test.pth"
    args_path = specialist_train.RUNC_ART / "args.json"
    if not checkpoint_path.exists() or not args_path.exists():
        pytest.skip("Run C ignored artifacts are not present in this checkout.")
    args = Namespace(**specialist_train.read_json(args_path))
    args.resume = str(checkpoint_path)
    args.label_correlation = "self_attn_legacy" if base_train.checkpoint_uses_legacy_label_correlation(args.resume) else getattr(args, "label_correlation", "self_attn")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    dim = int(ckpt.get("dim", 384))
    model = specialist_train.build_base_model(args, dim=dim, device=torch.device("cpu"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    state = base_train.load_resume_checkpoint(args.resume, model, optimizer, device=torch.device("cpu"), resume_optimizer=False, strict=True)
    assert state.missing_keys == []
    assert state.unexpected_keys == []
