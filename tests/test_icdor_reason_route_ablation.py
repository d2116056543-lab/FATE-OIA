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
    ).eval()


def test_reason_route_ablation_is_real_and_cannot_change_action_lane() -> None:
    model = _model()
    image = torch.randn(1, 3, 360, 640)
    model.set_factor_certificate_tiers(["reason_only"] * len(model.ontology["factors"]))
    # Production routes consume the preceding audit_visual state, never the
    # current batch's self-measurement. Seed that completed audit here so the
    # ablation exercises a real semantic route.
    model.continuous_credibility.update_from_audit(
        torch.full((len(model.ontology["factors"]),), 0.4)
    )
    full = model(image, route_mode="off", latent_enabled=True, reason_route_mode="full")
    off = model(image, route_mode="off", latent_enabled=True, reason_route_mode="off")
    shuffled = model(image, route_mode="off", latent_enabled=True, reason_route_mode="shuffled")

    assert torch.equal(full["action_final_logits"], off["action_final_logits"])
    assert torch.equal(full["action_final_logits"], shuffled["action_final_logits"])
    assert torch.count_nonzero(off["reason_factor_route_enabled_effective"]) == 0
    assert not torch.equal(full["reason_factor_router_weights"], shuffled["reason_factor_router_weights"])
