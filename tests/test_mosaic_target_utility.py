from __future__ import annotations

import pytest
import torch

from fate_oia.models.mosaic_native_semantics import load_icdor_ontology
from fate_oia.models.mosaic_target_sparse_router import MOSAICTargetSparseRouter


def _transfer_rows() -> list[dict[str, object]]:
    return [
        {
            "factor_id": "traffic_light_visible",
            "target_id": "action:forward",
            "available": True,
            "tes": 0.04,
            "tet": 0.03,
            "cca": 0.80,
            "ap_delta": 0.02,
        },
        {
            "factor_id": "traffic_light_visible",
            "target_id": "reason:reason_0",
            "available": True,
            "tes": 0.04,
            "tet": 0.03,
            "cca": 0.80,
            "ap_delta": 0.02,
        },
    ]


def test_target_utility_accepts_only_audit_target_evidence() -> None:
    module = __import__("fate_oia.engine.build_mosaic_target_utility", fromlist=["build_target_utility"])
    ontology = {
        "factors": [{"name": "traffic_light_visible"}],
        "action_names": ["forward", "stop", "left", "right"],
        "reason_names": ["reason_0"] + [f"reason_{index}" for index in range(1, 21)],
    }
    with pytest.raises(ValueError, match="audit_target"):
        module.build_target_utility({"source_split": "test", "per_target": _transfer_rows()}, ontology)

    utility = module.build_target_utility(
        {"source_split": "audit_target", "per_target": _transfer_rows()}, ontology
    )
    assert utility["source_split"] == "audit_target"
    assert utility["action_target_utility"][0][0] > 0.0
    assert utility["semantic_compatibility"][0][0] > 0.0
    assert utility["action_target_utility"][0][1] == 0.0


def test_target_utility_state_is_separate_for_reason_and_action() -> None:
    module = __import__("fate_oia.models.mosaic_target_utility", fromlist=["MOSAICAuditTargetUtility"])
    state = module.MOSAICAuditTargetUtility(factor_count=2, reason_count=21, action_count=4)
    state.update_from_audit(
        torch.full((21, 2), 0.25),
        torch.full((2, 4), 0.75),
        source_split="audit_target",
    )
    output = state()
    assert output["semantic_compatibility"].shape == (21, 2)
    assert output["action_target_utility"].shape == (2, 4)
    assert output["target_utility_initialized"].item() == 1
    with pytest.raises(ValueError, match="audit_target"):
        state.update_from_audit(torch.ones(21, 2), torch.ones(2, 4), source_split="test")


def test_action_router_uses_audit_utility_without_reason_state() -> None:
    torch.manual_seed(11)
    ontology = load_icdor_ontology("configs")
    router = MOSAICTargetSparseRouter(ontology, dim=16).eval()
    factor_count = len(ontology["factors"])
    features = torch.randn(1, factor_count, 16)
    queries = torch.randn(1, 4, 16)
    evidence = torch.ones(1, factor_count)
    baseline = router(features, evidence, evidence, queries, route_mode="shadow")
    utility = torch.full((factor_count, 4), 0.10)
    factor_index, action_index = router.candidate_edge_mask.any(dim=0).nonzero()[0].tolist()
    utility[factor_index, action_index] = 1.0
    routed = router(
        features,
        evidence,
        evidence,
        queries,
        route_mode="shadow",
        factor_target_utility=utility,
    )

    assert torch.equal(routed["action_target_utility_effective"], utility.unsqueeze(0))
    assert not torch.allclose(baseline["support_route_logits"], routed["support_route_logits"])
