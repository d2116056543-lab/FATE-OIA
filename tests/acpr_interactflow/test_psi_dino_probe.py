from __future__ import annotations

import torch


def test_dino_probe_model_mock_forward_single_and_clip_modes() -> None:
    from fate_oia.engine.train_psi_dino_probe import PSIDinoProbeModel

    single = PSIDinoProbeModel(
        input_mode="target_frame",
        selected_layers=(3, 7, 11),
        pretrained_weights="unused.pth",
        use_mock_dino=True,
        mock_dim=32,
        dino_input_size=(64, 96),
        dino_chunk_size=2,
    )
    clip = PSIDinoProbeModel(
        input_mode="k_current_15",
        selected_layers=(3, 7, 11),
        pretrained_weights="unused.pth",
        use_mock_dino=True,
        mock_dim=32,
        dino_input_size=(64, 96),
        dino_chunk_size=3,
    )

    out_single = single(torch.randn(2, 3, 48, 80))
    out_clip = clip(torch.randn(2, 15, 3, 48, 80))

    assert out_single["action_logits"].shape == (2, 3)
    assert out_single["exp29_logits"].shape == (2, 29)
    assert out_clip["action_logits"].shape == (2, 3)
    assert out_clip["exp29_logits"].shape == (2, 29)
    assert out_clip["probe_features"].shape[0] == 2
    assert all(not p.requires_grad for p in single.dino.parameters())


def test_dino_probe_temporal_attention_pooler_returns_frame_weights() -> None:
    from fate_oia.engine.train_psi_dino_probe import PSIDinoProbeModel

    model = PSIDinoProbeModel(
        input_mode="k_current_15",
        selected_layers=(3, 7, 11),
        pretrained_weights="unused.pth",
        use_mock_dino=True,
        mock_dim=32,
        dino_input_size=(64, 96),
        dino_chunk_size=5,
        temporal_pooler="attention",
    )

    out = model(torch.randn(2, 15, 3, 48, 80))

    assert out["action_logits"].shape == (2, 3)
    assert out["temporal_attention"].shape == (2, 15)
    assert torch.allclose(out["temporal_attention"].sum(dim=1), torch.ones(2), atol=1e-5)
    assert out["probe_features"].shape == (2, model.feature_dim * 4)


def test_dino_probe_manifest_records_protocol_and_no_cache() -> None:
    from fate_oia.engine.train_psi_dino_probe import build_probe_manifest

    manifest = build_probe_manifest(
        input_mode="k_current_15",
        package_root="pkg",
        frames_root="frames",
        protocol_index_dir="indices",
        protocol_name="gap_decay_180",
        exp_supervision_policy="near_keyframe_raw_mask",
        eval_exp_supervision_policy="near_keyframe_raw_mask",
        dino_weights="ckp/reference/dino_deitsmall8_pretrain.pth",
        selected_layers=(3, 7, 11),
        dino_input_size=(320, 576),
        temporal_pooler="attention",
        spatial_pooler="attention",
        spatial_queries=6,
        train_count=10,
        test_count=5,
        batch_size=4,
        num_workers=0,
        device="cpu",
        use_mock_dino=True,
        use_decision_group_weight=True,
        seed=123,
    )

    assert manifest["purpose"] == "frozen_dino_psi_learnability_probe"
    assert manifest["protocol_name"] == "gap_decay_180"
    assert manifest["input_mode"] == "k_current_15"
    assert manifest["feature_cache_enabled"] is False
    assert manifest["token_cache_enabled"] is False
    assert manifest["logit_cache_enabled"] is False
    assert manifest["dino_frozen"] is True
    assert manifest["selected_layers"] == [3, 7, 11]
    assert manifest["dino_input_size"] == [320, 576]
    assert manifest["temporal_pooler"] == "attention"
    assert manifest["spatial_pooler"] == "attention"
    assert manifest["spatial_queries"] == 6
    assert manifest["use_decision_group_weight"] is True
    assert manifest["seed"] == 123


def test_seed_helper_makes_probe_initialization_reproducible() -> None:
    from fate_oia.engine.train_psi_dino_probe import PSIDinoProbeModel, set_training_seed

    kwargs = {
        "input_mode": "target_frame",
        "selected_layers": (3, 7, 11),
        "pretrained_weights": "unused.pth",
        "use_mock_dino": True,
        "mock_dim": 32,
        "dino_input_size": (64, 96),
        "dino_chunk_size": 2,
        "spatial_pooler": "attention",
        "spatial_queries": 4,
        "hidden_dim": 64,
    }

    set_training_seed(123)
    model_a = PSIDinoProbeModel(**kwargs)
    weight_a = model_a.action_head.weight.detach().clone()

    set_training_seed(123)
    model_b = PSIDinoProbeModel(**kwargs)
    weight_b = model_b.action_head.weight.detach().clone()

    set_training_seed(124)
    model_c = PSIDinoProbeModel(**kwargs)
    weight_c = model_c.action_head.weight.detach().clone()

    assert torch.allclose(weight_a, weight_b)
    assert not torch.allclose(weight_a, weight_c)


def test_loader_generator_makes_shuffle_order_reproducible() -> None:
    from fate_oia.engine.train_psi_dino_probe import make_loader_generator

    gen_a = make_loader_generator(11)
    gen_b = make_loader_generator(11)
    gen_c = make_loader_generator(12)

    assert gen_a is not None
    assert gen_b is not None
    assert gen_c is not None
    order_a = torch.randperm(32, generator=gen_a)
    order_b = torch.randperm(32, generator=gen_b)
    order_c = torch.randperm(32, generator=gen_c)

    assert torch.equal(order_a, order_b)
    assert not torch.equal(order_a, order_c)


def test_exp_deploy_shift_best_locked_uses_resume_best_metric() -> None:
    from fate_oia.engine.train_psi_dino_probe import resolve_exp_deploy_shift

    resume_metrics = {
        "exp29_train_deploy_shift": 0.9060256183,
        "exp29_deploy_shift": 0.8929164111,
    }

    shift = resolve_exp_deploy_shift(
        mode="best_locked",
        fixed_shift=None,
        auto_shift=0.75,
        resume_metrics=resume_metrics,
    )

    assert shift == 0.9060256183


def test_regression_guard_stops_when_joint_falls_below_best() -> None:
    from fate_oia.engine.train_psi_dino_probe import should_stop_for_metric_regression

    assert should_stop_for_metric_regression(
        current_joint=0.3861926883,
        best_joint=0.3937390342,
        max_regression=0.003,
    )
    assert not should_stop_for_metric_regression(
        current_joint=0.3920,
        best_joint=0.3937390342,
        max_regression=0.003,
    )


def test_primary_metric_can_use_action_metric_for_action_first_runs() -> None:
    from fate_oia.engine.train_psi_dino_probe import get_primary_metric_value, should_stop_for_metric_regression

    current = {"joint": 0.2966, "Act_mAcc": 0.3877}
    best = {"joint": 0.3224, "Act_mAcc": 0.4228}

    assert get_primary_metric_value(current, "Act_mAcc") == 0.3877
    assert should_stop_for_metric_regression(
        current_metrics=current,
        best_metrics=best,
        primary_metric="Act_mAcc",
        max_regression=0.02,
    )
    assert not should_stop_for_metric_regression(
        current_metrics={"joint": 0.10, "Act_mAcc": 0.4210},
        best_metrics=best,
        primary_metric="Act_mAcc",
        max_regression=0.02,
    )


def test_best_split_selects_validation_metrics_without_using_test() -> None:
    from fate_oia.engine.train_psi_dino_probe import select_metrics_for_best_checkpoint

    test_metrics = {"split": "test", "Act_mAcc": 0.91, "joint": 0.80}
    val_metrics = {"split": "val", "Act_mAcc": 0.55, "joint": 0.50}

    selected = select_metrics_for_best_checkpoint(
        best_split="val",
        test_metrics=test_metrics,
        val_metrics=val_metrics,
    )

    assert selected is val_metrics
    assert selected["Act_mAcc"] == 0.55


def test_optimizer_restore_can_be_disabled_for_low_lr_continuation() -> None:
    from fate_oia.engine.train_psi_dino_probe import should_restore_resume_optimizer

    assert should_restore_resume_optimizer(checkpoint_has_optimizer=True, reset_optimizer_on_resume=False)
    assert not should_restore_resume_optimizer(checkpoint_has_optimizer=True, reset_optimizer_on_resume=True)
    assert not should_restore_resume_optimizer(checkpoint_has_optimizer=False, reset_optimizer_on_resume=False)


def test_action_rate_prior_loss_is_zero_when_prediction_rate_matches_target() -> None:
    from fate_oia.engine.train_psi_dino_probe import action_rate_prior_loss

    target_prior = torch.tensor([0.40, 0.50, 0.10], dtype=torch.float32)
    logits = target_prior.clamp_min(1e-6).log().view(1, 3).repeat(8, 1)

    loss, pred_rate = action_rate_prior_loss(logits, target_prior)

    assert loss.item() < 1e-6
    assert torch.allclose(pred_rate, target_prior, atol=1e-6)


def test_action_rate_prior_loss_penalizes_reduce_collapse() -> None:
    from fate_oia.engine.train_psi_dino_probe import action_rate_prior_loss

    target_prior = torch.tensor([0.40, 0.50, 0.10], dtype=torch.float32)
    collapsed_logits = torch.tensor([[-2.0, 3.0, -2.0]], dtype=torch.float32).repeat(16, 1)

    loss, pred_rate = action_rate_prior_loss(collapsed_logits, target_prior)

    assert pred_rate[1] > 0.95
    assert loss.item() > 0.25
