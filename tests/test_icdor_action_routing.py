from __future__ import annotations

from pathlib import Path
from copy import deepcopy

import torch
import pytest

from fate_oia.models.mosaic_icdor_action_decoder import MOSAICICDORActionDecoder
from fate_oia.models.mosaic_masked_target_rereader import MOSAICMaskedTargetRereader
from fate_oia.models.mosaic_native_semantics import load_icdor_ontology
from fate_oia.models.mosaic_target_sparse_router import MOSAICTargetSparseRouter


def _pyramid(batch_size: int, dim: int) -> dict[str, torch.Tensor]:
    return {
        "F_hi": torch.randn(batch_size, dim, 45, 80),
        "F_mid": torch.randn(batch_size, dim, 23, 40),
        "F_ctx": torch.randn(batch_size, dim, 12, 20),
    }


def test_icdor_router_has_exact_zero_disallowed_mass_and_dustbin_rejection() -> None:
    ontology = load_icdor_ontology(Path("configs"))
    router = MOSAICTargetSparseRouter(ontology, dim=32)
    factor_count = len(ontology["factors"])
    factor_features = torch.randn(2, factor_count, 32)
    evidence = torch.rand(2, factor_count)
    action_queries = torch.randn(2, 4, 32)

    router.set_certificate_tiers(["certified"] * factor_count)
    router.set_edge_admission(router.candidate_edge_mask)
    admitted = router(factor_features, evidence, 1.0 - evidence, action_queries, route_mode="admitted")
    disallowed = ~admitted["active_edge_mask"]
    for direction_index, direction in enumerate(("support", "veto")):
        weights = admitted[f"{direction}_weights"]
        assert torch.count_nonzero(weights.masked_select(disallowed[direction_index].unsqueeze(0))) == 0
        assert torch.allclose(
            weights.sum(dim=1) + admitted[f"{direction}_dustbin"],
            torch.ones_like(admitted[f"{direction}_dustbin"]),
            atol=1e-6,
        )

    off = router(factor_features, evidence, 1.0 - evidence, action_queries, route_mode="off")
    assert torch.count_nonzero(off["support_weights"]) == 0
    assert torch.count_nonzero(off["veto_weights"]) == 0
    assert torch.allclose(off["support_dustbin"], torch.ones_like(off["support_dustbin"]))
    assert torch.allclose(off["veto_dustbin"], torch.ones_like(off["veto_dustbin"]))


def test_icdor_action_visual_and_masked_reread_are_separate_and_mask_effective() -> None:
    torch.manual_seed(7)
    ontology = load_icdor_ontology(Path("configs"))
    factor_count = len(ontology["factors"])
    decoder = MOSAICICDORActionDecoder(dim=32, highres_topk=64, midres_topk=32)
    action = decoder(_pyramid(batch_size=2, dim=32))
    assert action["action_visual_logits"].shape == (2, 4)
    assert action["action_visual_attention"].shape == (2, 4, 45, 80)

    router = MOSAICTargetSparseRouter(ontology, dim=32)
    router.set_certificate_tiers(["certified"] * factor_count)
    router.set_edge_admission(router.candidate_edge_mask)
    route = router(
        torch.randn(2, factor_count, 32),
        torch.rand(2, factor_count),
        torch.rand(2, factor_count),
        action["action_queries"].detach(),
        route_mode="admitted",
    )
    rereader = MOSAICMaskedTargetRereader(dim=32, action_count=4, topk=64)
    rereader.set_gate_cap(0.05)
    masks = torch.rand(2, factor_count, 45, 80)
    action_pyramid = _pyramid(batch_size=2, dim=32)
    first = rereader(
        action_pyramid,
        action["action_queries"].detach(),
        masks.detach(),
        route["support_weights"],
        route["veto_weights"],
    )
    second = rereader(
        action_pyramid,
        action["action_queries"].detach(),
        masks.flip(-1).detach(),
        route["support_weights"],
        route["veto_weights"],
    )
    assert first["action_support_logits"].shape == (2, 4)
    assert first["action_veto_logits"].shape == (2, 4)
    assert float(first["action_route_gate_cap"]) == pytest.approx(0.05)
    assert float(first["action_support_gate"].max()) <= 0.05
    assert not torch.allclose(first["action_support_logits"], second["action_support_logits"])


def test_icdor_router_uses_edge_polarity_instead_of_direction_as_polarity() -> None:
    ontology = deepcopy(load_icdor_ontology(Path("configs")))
    action_name = next(iter(ontology["action_routes"]))
    edge = ontology["action_routes"][action_name]["support"][0]
    factor_id = ontology["factor_index"][edge["factor"]]
    action_id = ontology["action_index"][action_name]
    edge["polarity"] = "absent"
    router = MOSAICTargetSparseRouter(ontology, dim=32)
    router.set_certificate_tiers(["certified"] * router.factor_count)
    features = torch.randn(1, router.factor_count, 32)
    positive = torch.full((1, router.factor_count), 0.9)
    negative = torch.full((1, router.factor_count), 0.1)
    output = router(features, positive, negative, torch.randn(1, 4, 32), route_mode="shadow")
    assert output["active_edge_polarity_mask"][0, 1, factor_id, action_id]
    assert not output["active_edge_polarity_mask"][0, 0, factor_id, action_id]
    assert torch.allclose(output["support_route_evidence"][0, factor_id, action_id], negative[0, factor_id])
