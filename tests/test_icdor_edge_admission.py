from __future__ import annotations

from pathlib import Path

from fate_oia.engine.build_mosaic_edge_admission import (
    MOSAICEdgeInterventionStats,
    build_edge_admission,
)
from fate_oia.models.mosaic_native_semantics import load_icdor_ontology


def test_icdor_edge_admission_requires_all_audit_conditions_and_is_hash_stable() -> None:
    ontology = load_icdor_ontology(Path("configs"))
    factor_count = len(ontology["factors"])
    tiers = ["certified"] * factor_count
    key = ("support", "center_drivable_region_visible", "forward")
    stats = {
        key: MOSAICEdgeInterventionStats(
            valid_samples=80,
            signed_effect_lcb95=0.03,
            tet_lcb95=0.02,
            tes_lcb95=0.01,
            cca=0.70,
            isolated_edge_ap=0.71,
            visual_ap=0.70,
        )
    }
    first = build_edge_admission(stats, ontology, tiers, source_split="train_audit")
    second = build_edge_admission(stats, ontology, tiers, source_split="train_audit")
    assert first.sha256 == second.sha256
    factor_index = ontology["factor_index"]["center_drivable_region_visible"]
    action_index = ontology["action_index"]["forward"]
    assert bool(first.edge_admission_mask[0, factor_index, action_index])


def test_icdor_edge_admission_fails_closed_for_weak_or_reason_only_edges() -> None:
    ontology = load_icdor_ontology(Path("configs"))
    tiers = ["certified"] * len(ontology["factors"])
    tiers[ontology["factor_index"]["center_drivable_region_visible"]] = "reason_only"
    stats = {
        ("support", "center_drivable_region_visible", "forward"): MOSAICEdgeInterventionStats(
            valid_samples=63,
            signed_effect_lcb95=0.01,
            tet_lcb95=0.01,
            tes_lcb95=0.01,
            cca=0.59,
            isolated_edge_ap=0.60,
            visual_ap=0.70,
        )
    }
    admission = build_edge_admission(stats, ontology, tiers, source_split="train_audit")
    assert not bool(admission.edge_admission_mask.any())
