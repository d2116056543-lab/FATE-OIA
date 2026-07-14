from __future__ import annotations

from pathlib import Path

import torch

from fate_oia.models.acpr_mosaic_trust_icdor_model import MOSAICTrustICDORModel


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


def test_model_loads_only_a_complete_train_audit_certificate() -> None:
    model = _model()
    certificate = {
        "source_split": "train_audit",
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
        assert "train_audit" in str(error)
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
    assert torch.equal(query["factor_presence_prob"], full["factor_presence_prob"].roll(1, 1))
    assert not torch.allclose(full["factor_presence_prob"], image["factor_presence_prob"])


def test_auto_route_is_fail_closed_then_uses_frozen_admission() -> None:
    model = _model()
    images = torch.randn(1, 3, 360, 640)
    off = model(images)
    assert off["route_mode_code"].item() == 0
    model.set_factor_certificate_tiers(["certified"] * len(model.ontology["factors"]))
    shadow = model(images)
    assert shadow["route_mode_code"].item() == 1
    model.set_edge_admission(model.action_router.candidate_edge_mask)
    admitted = model(images)
    assert admitted["route_mode_code"].item() == 2


def test_reason_diagnostics_reuse_one_visual_forward_and_export_real_ablations() -> None:
    model = _model().eval()
    model.set_factor_certificate_tiers(["certified"] * len(model.ontology["factors"]))
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
            return_masks=False,
            factor_intervention_keep_mask=keep,
        )

    assert torch.equal(deleted["factor_intervention_keep_mask"], keep)
    assert torch.all(deleted["factor_positive_evidence"][:, 0] == 0)
    assert torch.all(deleted["factor_visibility_prob"][:, 0] == 0)
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
