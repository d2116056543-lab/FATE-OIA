from __future__ import annotations

import torch
import yaml
from PIL import Image

from fate_oia.datasets.diva_caf_oia_dataset import SimpleResizeToTensor
from fate_oia.engine.train_diva_caf_oia import parse_args, resolve_config
from fate_oia.losses.diva_caf_losses import asymmetric_loss_with_logits, diva_caf_loss
from fate_oia.models.caf_bilevel_routing import BiLevelFactorRouter
from fate_oia.models.caf_factor_auditor import CriticalFactorAuditor
from fate_oia.models.caf_reason_reliability import ReasonReliabilityGate
from fate_oia.models.diva_visual_mixture_gate import branch_safe_guarded_action


def test_dataset_transform_uses_imagenet_normalize_and_patch_grid():
    transform = SimpleResizeToTensor(360, 640)
    image = Image.new("RGB", (1280, 720), color=(0, 0, 0))
    tensor, meta = transform(image)
    assert tensor.shape == (3, 360, 640)
    assert meta["patch_grid"] == (45, 80)
    assert tensor.min() < 0.0
    assert tensor.max() <= 0.0


def test_yaml_config_overrides_default_training_and_model_values(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {"image_height": 360, "image_width": 640, "patch_size": 8, "no_feature_cache": True, "eval_splits": ["test"]},
                "model": {"dim": 384, "layer_indices": [3, 6, 9, 12], "delta_cap": 0.07, "reason_cap": 0.21},
                "backbone": {"pretrained_weights": "ckp/reference/dino_deitsmall8_pretrain.pth", "checkpoint_key": "teacher", "dino_variant": "vit_small"},
                "caf": {"factor_topk": 4, "factor_group_topk": 2},
                "training": {"epochs": 32, "batch_size": 4, "gradient_accumulation_steps": 8, "lr": 1e-4, "min_lr": 1e-5, "warmup_epochs": 3, "weight_decay": 0.01},
            }
        ),
        encoding="utf-8",
    )
    args = parse_args(["--config", str(cfg_path), "--output_dir", str(tmp_path / "out")])
    resolved = resolve_config(args)
    assert resolved["model"]["dim"] == 384
    assert resolved["training"]["epochs"] == 32
    assert resolved["backbone"]["pretrained_weights"]
    assert resolved["caf"]["factor_topk"] == 4
    assert resolved["data"]["no_feature_cache"] is True


def test_branch_safe_guarded_action_selects_fate_when_actor_hurts():
    z_fate = torch.tensor([[4.0, -4.0, -4.0, -4.0]])
    z_actor = torch.tensor([[-4.0, 4.0, -4.0, -4.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    guarded, stats = branch_safe_guarded_action(z_fate, z_actor, labels, tolerance=0.0)
    assert torch.allclose(guarded, z_fate)
    assert stats["guarded_source"] == "fate"


def test_factor_auditor_masks_evidence_not_delta_scaling():
    auditor = CriticalFactorAuditor()
    z_full = torch.zeros(2, 4)
    z_without_selected = torch.ones(2, 4)
    z_without_random = torch.zeros(2, 4) + 0.1
    y = torch.zeros(2, 4)
    result = auditor(z_actor_full=z_full, z_actor_without_selected=z_without_selected, z_actor_without_random=z_without_random, y_action=y)
    assert result["method"] == "action_gt_loss_drop_on_evidence_mask"
    assert result["drop_selected"] > result["drop_random"]
    assert "z_actor_without_selected" in result


def test_router_reliability_is_per_action_group_and_uses_exp_inputs():
    router = BiLevelFactorRouter(dim=8, action_dim=4, num_groups=3, factor_topk=2, group_topk=2)
    action_tokens = torch.randn(2, 4, 8)
    factor_tokens = torch.randn(2, 6, 8)
    group_ids = torch.tensor([[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
    exp_prior = torch.ones(2, 4, 6)
    exp_reliability = torch.full((2, 4, 6), 0.5)
    route = router(action_tokens, factor_tokens, group_ids, exp_prior=exp_prior, exp_reliability=exp_reliability)
    assert route["weak_exp_scores"].abs().sum() > 0
    assert router.faith_ema.shape == (4, 3)
    delta = torch.ones(4, 3)
    router.update_reliability(selected_vs_random_per_action_group=delta, help_delta_per_action_group=delta, hurt_delta_per_action_group=torch.zeros_like(delta))
    assert torch.all(router.help_ema > 0)


def test_reason_reliability_uses_factor_support_not_base_reason():
    gate = ReasonReliabilityGate(reason_dim=21)
    reason_logits = torch.zeros(2, 21)
    base_reason = torch.ones(2, 21) * 9.0
    low_support = torch.zeros(2, 21)
    high_support = torch.ones(2, 21)
    g_low = gate(reason_logits, factor_support=low_support, base_reason_logits=base_reason)
    g_high = gate(reason_logits, factor_support=high_support, base_reason_logits=base_reason)
    assert torch.all(g_high > g_low)


def test_diva_caf_loss_defaults_to_asl_main_losses():
    outputs = {
        "z_fate_action_logits": torch.zeros(2, 4, requires_grad=True),
        "z_eva_action_logits": torch.zeros(2, 4, requires_grad=True),
        "z_actor_action_logits": torch.zeros(2, 4, requires_grad=True),
        "base_reason_logits": torch.zeros(2, 21, requires_grad=True),
        "final_reason_logits": torch.zeros(2, 21, requires_grad=True),
        "visual_gate": torch.full((2, 4), 0.2, requires_grad=True),
        "gate_target": torch.ones(2, 4),
        "selected_vs_random_stats": {"loss": torch.tensor(0.1, requires_grad=True)},
    }
    y_action = torch.ones(2, 4)
    y_reason = torch.zeros(2, 21)
    loss, terms = diva_caf_loss(outputs, y_action, y_reason)
    assert terms["loss_type"] == "asl"
    assert torch.isclose(terms["loss_action_actor"], asymmetric_loss_with_logits(outputs["z_actor_action_logits"], y_action))
    assert loss.requires_grad
