from __future__ import annotations

from pathlib import Path

import torch

from fate_oia.losses.mosaic_icdor_action_losses import action_route_losses
from fate_oia.models.acpr_mosaic_trust_icdor_model import MOSAICTrustICDORModel
from fate_oia.engine.train_acpr_mosaic_trust_icdor import build_icdor_model, load_config


def _model() -> MOSAICTrustICDORModel:
    return MOSAICTrustICDORModel(
        config_root=Path("configs"),
        use_mock_dino=True,
        mock_dim=32,
        adapter_rank=16,
        highres_topk=64,
        midres_topk=32,
        point_samples=4,
        curve_samples=16,
        region_samples=12,
    )


def test_icdor_model_uses_direct_image_three_lanes_and_action_firewall() -> None:
    model = _model()
    images = torch.randn(1, 3, 360, 640)
    outputs = model(images, route_mode="off", latent_enabled=False, return_masks=True)

    assert outputs["action_visual_logits"].shape == (1, 4)
    assert outputs["action_final_logits"].shape == (1, 4)
    assert torch.equal(outputs["action_final_logits"], outputs["action_visual_logits"])
    assert outputs["reason_visual_observed_logits"].shape == (1, 21)
    assert outputs["reason_observed_logits"].shape == (1, 21)
    assert outputs["factor_presence_prob"].shape[1] >= 20
    assert outputs["action_logits_deploy"].shape == (1, 4)
    assert outputs["reason_logits_deploy"].shape == (1, 21)
    assert not hasattr(model, "state_composer")
    assert model.factor_adapter is not model.action_adapter
    assert model.action_adapter is not model.reason_adapter

    model.zero_grad(set_to_none=True)
    outputs["action_final_logits"].sum().backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0 for parameter in model.action_adapter.parameters())
    assert all(parameter.grad is None for parameter in model.reason_adapter.parameters())
    assert all(parameter.grad is None for parameter in model.factor_adapter.parameters())


def test_icdor_model_admitted_route_is_action_visual_reread_not_factor_logit_delta() -> None:
    model = _model()
    factor_count = len(model.ontology["factors"])
    model.set_factor_certificate_tiers(["certified"] * factor_count)
    model.set_edge_admission(model.action_router.candidate_edge_mask)
    outputs = model(torch.randn(1, 3, 360, 640), route_mode="admitted", latent_enabled=True, return_masks=True)

    assert torch.allclose(
        outputs["action_shadow_logits"],
        outputs["action_visual_logits"] + outputs["action_support_logits"] - outputs["action_veto_logits"],
    )
    assert torch.equal(outputs["action_final_logits"], outputs["action_shadow_logits"])
    assert outputs["reason_factor_router_weights"].shape[1:] == (21, factor_count)
    assert outputs["action_matched_random_logits"].shape == (1, 4)
    assert torch.allclose(
        outputs["equal_mass_random_factor_masks"].sum((-2, -1)),
        outputs["factor_soft_masks"].sum((-2, -1)),
        atol=1e-5,
    )
    # Selected and matched-random arms must both use typed re-reading. A
    # coarse fallback makes a deletion comparison scientifically invalid.
    assert outputs["action_matched_random_typed_target_coordinates"].shape == (1, 4, 2)
    assert not torch.allclose(
        outputs["sampling_coordinates"],
        outputs["action_matched_random_sampling_coordinates"],
    )


def test_action_shadow_gradient_cannot_update_the_direct_action_visual_owner() -> None:
    """CREDO defines shadow logits as stopgrad(visual) plus the route delta."""
    model = _model()
    factor_count = len(model.ontology["factors"])
    model.continuous_credibility.update_from_audit(torch.full((factor_count,), 0.4))
    outputs = model(torch.randn(1, 3, 360, 640), route_mode="shadow", latent_enabled=True)
    model.zero_grad(set_to_none=True)
    outputs["action_shadow_logits"].sum().backward()

    assert all(parameter.grad is None for parameter in model.action_visual_decoder.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.action_router.parameters()
    )


def test_matched_random_route_loss_cannot_update_direct_action_visual_owner() -> None:
    """Matched controls train the route owner without bypassing CREDO's action firewall."""
    model = _model()
    factor_count = len(model.ontology["factors"])
    model.continuous_credibility.update_from_audit(torch.full((factor_count,), 0.4))
    outputs = model(torch.randn(2, 3, 360, 640), route_mode="shadow", latent_enabled=True)
    route = action_route_losses(
        outputs["action_visual_logits"],
        outputs["action_support_logits"],
        outputs["action_veto_logits"],
        torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]),
        support_dustbin=outputs["support_dustbin"],
        veto_dustbin=outputs["veto_dustbin"],
        pareto_penalty=outputs["action_shadow_logits"].sum() * 0.0,
        matched_random_logits=outputs["action_matched_random_logits"],
        intervention_weight=0.10,
    )
    model.zero_grad(set_to_none=True)
    route["loss_action_route_total"].backward()

    assert all(parameter.grad is None for parameter in model.action_visual_decoder.parameters())
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in model.action_router.parameters()
    )


def test_icdor_model_exposes_frozen_certificate_reliability_without_a_per_image_tier_head() -> None:
    model = _model()
    factor_count = len(model.ontology["factors"])
    tiers = ["reason_only"] * factor_count
    tiers[0] = "certified"
    tiers[1] = "abstained"
    model.set_factor_certificate_tiers(tiers)
    outputs = model(torch.randn(1, 3, 360, 640), route_mode="off", latent_enabled=True)

    assert outputs["factor_certificate_reliability"].tolist()[:2] == [1.0, 0.0]
    assert outputs["factor_certificate_reliability"].tolist()[2] == 0.5
    assert not any("tier" in name or "reliability" in name for name, _ in model.named_parameters())


def test_model_routes_with_previous_audit_visual_credibility() -> None:
    model = _model().eval()
    factor_count = len(model.ontology["factors"])
    previous_epoch = torch.linspace(0.05, 0.95, factor_count)
    model.continuous_credibility.update_from_audit(previous_epoch)

    with torch.no_grad():
        outputs = model(torch.randn(1, 3, 360, 640), route_mode="shadow", latent_enabled=True)

    expected = model.continuous_credibility.ema_cV.view(1, -1)
    assert torch.equal(outputs["cV"], expected)
    assert torch.equal(outputs["reason_continuous_credibility"], expected)


def test_model_consumes_disjoint_audit_target_state_in_action_and_reason_routes() -> None:
    model = _model().eval()
    factor_count = len(model.ontology["factors"])
    model.set_factor_certificate_tiers(["reason_only"] * factor_count)
    model.continuous_credibility.update_from_audit(torch.full((factor_count,), 0.4))
    image = torch.randn(1, 3, 360, 640)
    with torch.no_grad():
        before = model(image, route_mode="shadow", latent_enabled=True)

    semantic = torch.full((21, factor_count), 0.10)
    semantic[:, 0] = 1.0
    action = torch.full((factor_count, 4), 0.10)
    action[0] = 1.0
    model.target_utility.update_from_audit(semantic, action, source_split="audit_target")
    with torch.no_grad():
        after = model(image, route_mode="shadow", latent_enabled=True)

    assert torch.equal(after["semantic_compatibility"], semantic)
    assert torch.equal(after["action_target_utility"], action)
    assert torch.equal(after["action_target_utility_effective"], action.unsqueeze(0))
    assert torch.equal(after["reason_semantic_compatibility_effective"], semantic)
    assert not torch.allclose(before["reason_factor_router_weights"], after["reason_factor_router_weights"])


def test_model_loads_only_a_complete_train_audit_certificate() -> None:
    model = _model()
    certificate = {
        "source_split": "audit_visual",
        "sha256": "AB" * 32,
        "entries": {
            factor["name"]: {"tier": "certified", "reliability": 1.0}
            for factor in model.ontology["factors"]
        },
    }
    model.load_factor_certificate(certificate)

    assert model.certificate_sha256 == "AB" * 32
    assert torch.equal(model.factor_certificate_reliability, torch.ones_like(model.factor_certificate_reliability))
    certificate["source_split"] = "test"
    try:
        model.load_factor_certificate(certificate)
    except ValueError as error:
        assert "audit_visual" in str(error)
    else:
        raise AssertionError("test-derived certificate must be rejected")


def test_latent_reason_loss_does_not_update_reason_visual_adapter() -> None:
    model = _model()
    model.set_factor_certificate_tiers(["certified"] * len(model.ontology["factors"]))
    outputs = model(torch.randn(1, 3, 360, 640), route_mode="off", latent_enabled=True)
    model.zero_grad(set_to_none=True)
    outputs["reason_logits_latent"].sum().backward()

    assert all(parameter.grad is None for parameter in model.reason_adapter.parameters())
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0 for parameter in model.reason_latent_decoder.parameters())


def test_observed_reason_loss_does_not_update_factor_or_action_lanes() -> None:
    model = _model()
    model.set_factor_certificate_tiers(["certified"] * len(model.ontology["factors"]))
    outputs = model(torch.randn(1, 3, 360, 640), route_mode="off", latent_enabled=True)
    model.zero_grad(set_to_none=True)
    outputs["reason_observed_logits"].sum().backward()

    assert all(parameter.grad is None for parameter in model.factor_adapter.parameters())
    assert all(parameter.grad is None for parameter in model.action_adapter.parameters())


def test_factor_ablation_modes_execute_real_distinct_forwards() -> None:
    torch.manual_seed(12)
    model = _model().eval()
    images = torch.randn(2, 3, 360, 640)
    with torch.no_grad():
        full = model(images, return_masks=True, factor_ablation_mode="full")
        content = model(images, return_masks=True, factor_ablation_mode="content_only")
        prior = model(images, return_masks=True, factor_ablation_mode="prior_only")
        query = model(images, return_masks=True, factor_ablation_mode="query_shuffled")
        image = model(images, return_masks=True, factor_ablation_mode="image_shuffled")
    assert not torch.allclose(full["factor_presence_prob"], content["factor_presence_prob"])
    assert not torch.allclose(full["factor_presence_prob"], prior["factor_presence_prob"])
    assert not torch.equal(query["factor_presence_prob"], full["factor_presence_prob"].roll(1, 1))
    assert not torch.allclose(full["factor_presence_prob"], image["factor_presence_prob"])


def test_auto_route_uses_safe_shadow_before_frozen_admission() -> None:
    model = _model()
    images = torch.randn(1, 3, 360, 640)
    shadow = model(images)
    assert shadow["route_mode_code"].item() == 1
    assert torch.allclose(shadow["action_final_logits"], shadow["action_visual_logits"])
    model.set_factor_certificate_tiers(["certified"] * len(model.ontology["factors"]))
    still_shadow = model(images)
    assert still_shadow["route_mode_code"].item() == 1
    model.set_edge_admission(model.action_router.candidate_edge_mask)
    admitted = model(images)
    assert admitted["route_mode_code"].item() == 2


def test_reason_diagnostics_reuse_one_visual_forward_and_export_real_ablations() -> None:
    model = _model().eval()
    model.set_factor_certificate_tiers(["certified"] * len(model.ontology["factors"]))
    # Production routes consume the preceding audit_visual result.  Seed a
    # nonzero audited state here so this test exercises real interventions,
    # rather than the intentionally safe zero-credibility bootstrap route.
    model.continuous_credibility.update_from_audit(
        torch.full((len(model.ontology["factors"]),), 0.4)
    )
    with torch.no_grad():
        output = model(torch.randn(2, 3, 360, 640), latent_enabled=True, return_diagnostics=True)
    assert output["reason_observed_logits_route_off"].shape == (2, 21)
    assert output["reason_observed_logits_route_shuffled"].shape == (2, 21)
    assert not torch.allclose(output["reason_observed_logits"], output["reason_observed_logits_route_shuffled"])
    for key in (
        "action_factor_off_logits", "action_factor_shuffled_logits",
        "action_wrong_target_logits", "action_equal_mass_random_logits",
    ):
        assert output[key].shape == (2, 4)


def test_factor_deletion_intervention_is_real_and_label_free() -> None:
    model = _model().eval()
    factor_count = len(model.ontology["factors"])
    model.set_factor_certificate_tiers(["certified"] * factor_count)
    model.set_edge_admission(model.action_router.candidate_edge_mask)
    images = torch.randn(2, 3, 360, 640)
    keep = torch.ones(2, factor_count)
    keep[:, 0] = 0.0

    with torch.no_grad():
        full = model(images, route_mode="admitted", latent_enabled=True, return_masks=False)
        deleted = model(
            images,
            route_mode="admitted",
            latent_enabled=True,
            return_masks=True,
            factor_intervention_keep_mask=keep,
        )

    assert torch.equal(deleted["factor_intervention_keep_mask"], keep)
    assert torch.all(deleted["factor_positive_evidence"][:, 0] == 0)
    assert torch.all(deleted["factor_visibility_prob"][:, 0] == 0)
    # A deleted factor must not survive through the fine typed rereader.
    assert torch.all(deleted["sample_attention"][:, 0] == 0)
    assert not torch.allclose(full["reason_observed_logits"], deleted["reason_observed_logits"])


def test_batch_local_dino_reuse_is_exact_and_not_a_persistent_cache() -> None:
    model = _model().eval()
    images = torch.randn(2, 3, 360, 640)
    with torch.no_grad():
        field = model.dino(images)
        direct = model(images, return_masks=False)
        reused = model(images, return_masks=False, precomputed_dino_field=field)
    assert torch.equal(direct["action_final_logits"], reused["action_final_logits"])
    assert torch.equal(direct["reason_observed_logits"], reused["reason_observed_logits"])
    assert not hasattr(model, "feature_cache")


def test_target_controls_reuse_only_batch_local_static_state() -> None:
    """A matched control may rerun target reads, never its static visual measurement."""
    model = _model().eval()
    factor_count = len(model.ontology["factors"])
    model.continuous_credibility.update_from_audit(torch.full((factor_count,), 0.4))
    images = torch.randn(1, 3, 360, 640)
    calls = {"factor_pyramid": 0, "action_pyramid": 0, "reason_pyramid": 0, "factor_extractor": 0, "router": 0, "rereader": 0}
    hooks = [
        model.factor_visual_pyramid.register_forward_hook(lambda *_: calls.__setitem__("factor_pyramid", calls["factor_pyramid"] + 1)),
        model.action_visual_pyramid.register_forward_hook(lambda *_: calls.__setitem__("action_pyramid", calls["action_pyramid"] + 1)),
        model.reason_visual_pyramid.register_forward_hook(lambda *_: calls.__setitem__("reason_pyramid", calls["reason_pyramid"] + 1)),
        model.factor_extractor.register_forward_hook(lambda *_: calls.__setitem__("factor_extractor", calls["factor_extractor"] + 1)),
        model.action_router.register_forward_hook(lambda *_: calls.__setitem__("router", calls["router"] + 1)),
        model.action_rereader.register_forward_hook(lambda *_: calls.__setitem__("rereader", calls["rereader"] + 1)),
    ]
    try:
        with torch.no_grad():
            field = model.dino(images)
            direct = model(images, route_mode="shadow", latent_enabled=True, precomputed_dino_field=field, return_masks=True)
            context = model.prepare_intervention_context(images, precomputed_dino_field=field)
            static_after_context = {key: calls[key] for key in ("factor_pyramid", "action_pyramid", "reason_pyramid", "factor_extractor")}
            contextual = model(images, route_mode="shadow", latent_enabled=True, intervention_context=context, return_masks=True)
            override = direct["factor_soft_masks"].clone()
            override[:, 0] = 0.0
            override[0, 0, 8, 12] = 1.0
            controlled = model(
                images,
                route_mode="shadow",
                latent_enabled=True,
                intervention_context=context,
                factor_mask_override=override,
            )
    finally:
        for hook in hooks:
            hook.remove()

    assert torch.equal(contextual["action_final_logits"], direct["action_final_logits"])
    assert torch.equal(contextual["reason_observed_logits"], direct["reason_observed_logits"])
    assert {key: calls[key] for key in static_after_context} == static_after_context
    assert calls["router"] >= 3
    assert calls["rereader"] >= 3
    assert controlled["factor_override_recomputed_typed_coordinates"].item() == 1.0


def test_factor_audit_context_matches_full_outputs_without_target_decoders() -> None:
    """Factor credibility ablations must not replay unrelated target branches."""
    torch.manual_seed(67)
    model = _model().eval()
    images = torch.randn(2, 3, 360, 640)
    modes = ("full", "content_only", "prior_only", "query_shuffled", "image_shuffled")
    with torch.no_grad():
        direct = {
            mode: model(images, factor_ablation_mode=mode, return_masks=True)
            for mode in modes
        }

    calls = {"dino": 0, "factor": 0, "action": 0, "reason": 0, "extractor": 0}
    hooks = [
        model.dino.register_forward_hook(lambda *_: calls.__setitem__("dino", calls["dino"] + 1)),
        model.factor_visual_pyramid.register_forward_hook(lambda *_: calls.__setitem__("factor", calls["factor"] + 1)),
        model.action_visual_pyramid.register_forward_hook(lambda *_: calls.__setitem__("action", calls["action"] + 1)),
        model.reason_visual_pyramid.register_forward_hook(lambda *_: calls.__setitem__("reason", calls["reason"] + 1)),
        model.factor_extractor.register_forward_hook(lambda *_: calls.__setitem__("extractor", calls["extractor"] + 1)),
    ]
    try:
        # A distinct tensor identity ensures this test counts the audit's own
        # batch-local DINO field rather than reusing the direct baseline field.
        audit_images = images.clone()
        with torch.no_grad():
            context = model.prepare_factor_audit_context(audit_images)
            factor_only = {
                mode: model.forward_factor_audit(
                    audit_images,
                    factor_ablation_mode=mode,
                    context=context,
                )
                for mode in modes
            }
    finally:
        for hook in hooks:
            hook.remove()

    for mode in modes:
        for key in (
            "factor_presence_prob",
            "factor_visibility_prob",
            "factor_soft_masks",
            "prototype_weights",
        ):
            torch.testing.assert_close(factor_only[mode][key], direct[mode][key])
    assert calls == {"dino": 1, "factor": 1, "action": 0, "reason": 0, "extractor": 5}


def test_credibility_and_fine_transport_config_change_the_real_forward_path() -> None:
    config = load_config("configs/fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml")
    config["model"]["action_route"].update(
        {
            "action_shadow_credibility_floor": 0.07,
            "reason_semantic_credibility_floor": 0.19,
        }
    )
    config["credibility"].update(
        {
            "ema_decay": 0.61,
            "image_only_cap": 0.07,
            "unknown_cap": 0.0,
            "no_reliable_negative_cap": 0.19,
        }
    )
    config["fine_transport"].update(
        {
            "enabled": False,
            "point_eta": 0.41,
            "curve_eta": 0.52,
            "region_eta": 0.63,
            "local_reread_offset_max": 0.04,
            "fine_off_diagnostic": True,
            "coarse_off_diagnostic": True,
        }
    )
    model = build_icdor_model(config, use_mock_dino=True, mock_dim=32)

    assert model.credibility_independent_of_reason_labels is True
    assert model.action_shadow_credibility_floor == 0.07
    assert model.reason_semantic_credibility_floor == 0.19
    assert model.continuous_credibility.ema_decay == 0.61
    assert torch.isclose(model.continuous_credibility.factor_credibility_cap.max(), torch.tensor(1.0))
    assert model.factor_extractor.fine_transport_enabled is False
    assert model.factor_extractor.fine_eta_by_type == {
        "point": 0.41,
        "object": 0.41,
        "curve": 0.52,
        "region": 0.63,
    }
    assert model.action_rereader.typed_rereader.max_local_offset == 0.04
    assert model.reason_latent_decoder.typed_rereader.max_local_offset == 0.04
    assert model.fine_transport_diagnostics == {"fine_off": True, "coarse_off": True}

    output = model(torch.randn(1, 3, 360, 640), return_masks=True)
    assert torch.allclose(output["factor_soft_masks"], output["factor_coarse_masks"])


def test_override_mask_rebuilds_typed_coordinates_from_moved_evidence() -> None:
    masks = torch.zeros(1, 1, 5, 7)
    masks[0, 0, 1, 2] = 0.7
    masks[0, 0, 3, 5] = 0.9
    template = torch.zeros(1, 1, 1, 2, 3, 2)
    coordinates = MOSAICTrustICDORModel._typed_coordinates_from_override_mask(masks, template)
    x = (((coordinates[..., 0] + 1.0) * 7 / 2.0) - 0.5).round().long()
    y = (((coordinates[..., 1] + 1.0) * 5 / 2.0) - 0.5).round().long()
    sampled = masks[0, 0, y, x]
    assert torch.all(sampled > 0.0)
    # The highest-valued moved patch is the first typed slot.
    assert (int(y.flatten()[0]), int(x.flatten()[0])) == (3, 5)


def test_forward_override_moves_factor_typed_reread_coordinates() -> None:
    model = _model().eval()
    images = torch.randn(1, 3, 360, 640)
    with torch.no_grad():
        field = model.dino(images)
        baseline = model(images, return_masks=True, precomputed_dino_field=field)
        override = baseline["factor_soft_masks"].clone()
        override[:, 0] = 0.0
        override[0, 0, 11, 17] = 0.9
        moved = model(
            images,
            return_masks=True,
            precomputed_dino_field=field,
            factor_mask_override=override,
        )
    coordinates = moved["sampling_coordinates"][0, 0]
    x = (((coordinates[..., 0] + 1.0) * 80 / 2.0) - 0.5).round().long()
    y = (((coordinates[..., 1] + 1.0) * 45 / 2.0) - 0.5).round().long()
    assert torch.all(override[0, 0, y, x] > 0.0)
    assert moved["factor_override_recomputed_typed_coordinates"].item() == 1.0
