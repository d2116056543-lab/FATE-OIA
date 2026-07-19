from __future__ import annotations

import inspect

import torch

from fate_oia.engine.build_mosaic_edge_admission import (
    MOSAICEdgeInterventionStats,
    build_edge_admission,
)
from fate_oia.models.acpr_mosaic_trust_icdor_model import MOSAICTrustICDORModel

from _mosaic_v5_helpers import tiny_ontology


def _positive_stats() -> MOSAICEdgeInterventionStats:
    return MOSAICEdgeInterventionStats(
        valid_samples=64,
        signed_effect_lcb95=0.01,
        tet_lcb95=0.01,
        tes_lcb95=0.01,
        tes_identity_lcb95=0.01,
        tes_spatial_lcb95=0.01,
        cca=0.61,
        isolated_edge_ap=0.70,
        visual_ap=0.70,
    )


def test_target_effect_admits_edge_without_global_cv_threshold() -> None:
    ontology = tiny_ontology()
    statistics = {
        (direction, factor["name"], action): _positive_stats()
        for direction in ("support", "veto")
        for factor in ontology["factors"]
        for action in ontology["action_names"]
    }
    admission = build_edge_admission(
        statistics,
        ontology,
        factor_credibility=[0.0, 0.0, 0.0],
        source_split="audit_target",
    )
    # The tiny ontology offers factor_0 as support and factor_1 as veto. Both
    # target-effect-proven routes must pass even though the cV diagnostic is 0.
    assert torch.all(admission.edge_admission_mask[0, 0])
    assert torch.all(admission.edge_admission_mask[1, 1])
    assert not torch.any(admission.edge_admission_mask[0, 1:])
    assert not torch.any(admission.edge_admission_mask[1, (0, 2)])


def test_model_keeps_train_access_floor_separate_from_final_admission() -> None:
    init_signature = inspect.signature(MOSAICTrustICDORModel.__init__)
    forward_source = inspect.getsource(MOSAICTrustICDORModel.forward)
    assert "action_shadow_credibility_floor" in init_signature.parameters
    assert "reason_semantic_credibility_floor" in init_signature.parameters
    assert "action_credibility_min_for_admission" not in init_signature.parameters
    assert "action_route_train_access" in forward_source
    assert "reason_route_train_access" in forward_source
