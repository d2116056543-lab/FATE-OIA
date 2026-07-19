from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch

from fate_oia.models.mosaic_action_route_policy import partial_action_admission


@dataclass(frozen=True)
class MOSAICEdgeInterventionStats:
    valid_samples: int
    signed_effect_lcb95: float
    tet_lcb95: float
    tes_lcb95: float
    tes_identity_lcb95: float
    tes_spatial_lcb95: float
    cca: float
    isolated_edge_ap: float
    visual_ap: float
    calibration_gain: float | None = None


@dataclass(frozen=True)
class MOSAICEdgeAdmission:
    edge_admission_mask: torch.Tensor
    entries: dict[str, dict[str, Any]]
    sha256: str
    source_split: str = "audit_target"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "icdor_v3",
            "source_split": self.source_split,
            "entries": self.entries,
            "sha256": self.sha256,
        }


def _edge_key(direction: str, factor_name: str, action_name: str) -> str:
    return f"{direction}:{factor_name}:{action_name}"


def build_edge_admission(
    statistics: Mapping[tuple[str, str, str], MOSAICEdgeInterventionStats],
    ontology: Mapping[str, Any],
    factor_credibility: Sequence[float],
    *,
    source_split: str,
) -> MOSAICEdgeAdmission:
    """Admit only independent-audit-proven factor-to-action edges.

    A discrete factor certificate and continuous cV are reporting diagnostics.
    Neither is a precondition for target-specific action admission: applying a
    global cV cutoff recreates CREDO-MAP's cold-start loop.  Admission is based
    on this action's independently measured TET/TES/CCA/AP evidence.
    """
    if source_split not in {"audit_target", "train_audit"}:
        raise ValueError("IC-DOR edge admission may only read an independent target audit")
    factor_names = [factor["name"] for factor in ontology["factors"]]
    action_names = list(ontology["action_names"])
    if len(factor_credibility) != len(factor_names):
        raise ValueError("IC-DOR edge admission credibility does not match the factor ontology")
    credibility = torch.tensor(factor_credibility, dtype=torch.float32)
    if not torch.isfinite(credibility).all() or bool(((credibility < 0.0) | (credibility > 1.0)).any()):
        raise ValueError("IC-DOR edge admission credibility must be finite in [0,1]")
    factor_index = ontology["factor_index"]
    action_index = ontology["action_index"]
    candidate = torch.zeros(2, len(factor_names), len(action_names), dtype=torch.bool)
    for direction_id, direction in enumerate(("support", "veto")):
        for action_name, directions in ontology["action_routes"].items():
            for edge in directions[direction]:
                candidate[direction_id, factor_index[edge["factor"]], action_index[action_name]] = True
    # Aggregate independent target-audit evidence per action before accepting
    # any edge. This is intentionally partial: one action may be admitted
    # while another remains in shadow.
    action_tet = torch.zeros(len(action_names), dtype=torch.float32)
    action_tes = torch.zeros(len(action_names), dtype=torch.float32)
    action_cca = torch.zeros(len(action_names), dtype=torch.float32)
    action_visual_ap = torch.zeros(len(action_names), dtype=torch.float32)
    action_edge_ap = torch.zeros(len(action_names), dtype=torch.float32)
    for action_id, action_name in enumerate(action_names):
        for direction_id, direction in enumerate(("support", "veto")):
            for factor_id, factor_name in enumerate(factor_names):
                if not candidate[direction_id, factor_id, action_id]:
                    continue
                stats = statistics.get((direction, factor_name, action_name))
                if stats is None:
                    continue
                action_tet[action_id] = max(action_tet[action_id], float(stats.tet_lcb95))
                action_tes[action_id] = max(action_tes[action_id], float(stats.tes_lcb95))
                action_cca[action_id] = max(action_cca[action_id], float(stats.cca))
                action_visual_ap[action_id] = max(action_visual_ap[action_id], float(stats.visual_ap))
                action_edge_ap[action_id] = max(action_edge_ap[action_id], float(stats.isolated_edge_ap))
    partial_action_ready = partial_action_admission(
        action_tet,
        action_tes,
        action_cca,
        action_visual_ap,
        action_edge_ap,
    )
    admission = torch.zeros_like(candidate)
    entries: dict[str, dict[str, Any]] = {}
    for direction_id, direction in enumerate(("support", "veto")):
        for factor_id, factor_name in enumerate(factor_names):
            for action_id, action_name in enumerate(action_names):
                if not candidate[direction_id, factor_id, action_id]:
                    continue
                key = (direction, factor_name, action_name)
                stats = statistics.get(key)
                accepted = False
                reasons: list[str] = []
                if not bool(partial_action_ready[action_id]):
                    reasons.append("partial_action_admission_not_ready")
                if stats is None:
                    reasons.append("missing_target_audit_intervention")
                else:
                    if stats.valid_samples < 64:
                        reasons.append("insufficient_valid_interventions")
                    if stats.signed_effect_lcb95 <= 0:
                        reasons.append("signed_effect_lcb_not_positive")
                    if stats.tet_lcb95 <= 0:
                        reasons.append("tet_lcb_not_positive")
                    if stats.tes_lcb95 <= 0:
                        reasons.append("tes_lcb_not_positive")
                    if stats.tes_identity_lcb95 <= 0:
                        reasons.append("tes_identity_lcb_not_positive")
                    if stats.tes_spatial_lcb95 <= 0:
                        reasons.append("tes_spatial_lcb_not_positive")
                    if stats.cca < 0.60:
                        reasons.append("cca_below_0p60")
                    if stats.calibration_gain is not None and stats.calibration_gain <= 0.0:
                        reasons.append("calibration_gain_not_positive")
                    if stats.isolated_edge_ap < stats.visual_ap - 0.002:
                        reasons.append("isolated_edge_ap_harms_visual")
                    accepted = not reasons
                admission[direction_id, factor_id, action_id] = accepted
                entries[_edge_key(direction, factor_name, action_name)] = {
                    "direction": direction,
                    "factor": factor_name,
                    "target": action_name,
                    "candidate": True,
                    "factor_credibility": float(credibility[factor_id]),
                    "accepted": accepted,
                    "partial_action_ready": bool(partial_action_ready[action_id]),
                    "reasons": reasons,
                    "metrics": asdict(stats) if stats is not None else None,
                }
    unsigned = {"version": "icdor_v3", "source_split": source_split, "entries": entries}
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest().upper()
    return MOSAICEdgeAdmission(
        edge_admission_mask=admission,
        entries=entries,
        sha256=digest,
        source_split=source_split,
    )
