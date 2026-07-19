from __future__ import annotations

import torch

from fate_oia.models.mosaic_target_sparse_router import MOSAICTargetSparseRouter
from _mosaic_v5_helpers import tiny_ontology


def test_route_mass_monotonic_with_credibility() -> None:
    router = MOSAICTargetSparseRouter(tiny_ontology(), dim=8)
    features = torch.ones(1, 3, 8)
    evidence = torch.ones(1, 3)
    queries = torch.ones(1, 4, 8)
    low = router(features, evidence, evidence, queries, route_mode="shadow", factor_credibility=torch.full((1, 3), 0.10))
    high = router(features, evidence, evidence, queries, route_mode="shadow", factor_credibility=torch.ones(1, 3))
    assert torch.all(high["support_route_mass"] >= low["support_route_mass"])
    assert torch.allclose(high["support_weights"].sum(1), high["support_route_mass"], atol=1e-7)
