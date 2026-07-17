from __future__ import annotations

from pathlib import Path

import torch

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


def test_credibility_and_fine_transport_config_change_the_real_forward_path() -> None:
    config = load_config("configs/fate_oia_train_360x640_acpr_mosaic_trust_v3_icdor.yaml")
    config["credibility"].update(
        {
            "observable_cV_min_for_admission": 0.73,
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
    assert model.action_credibility_min_for_admission == 0.73
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
