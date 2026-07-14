from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import torch


@dataclass(frozen=True)
class MOSAICEdgeInterventionStats:
    valid_samples: int
    signed_effect_lcb95: float
    tet_lcb95: float
    tes_lcb95: float
    cca: float
    isolated_edge_ap: float
    visual_ap: float
    calibration_gain: float | None = None


@dataclass(frozen=True)
class MOSAICEdgeAdmission:
    edge_admission_mask: torch.Tensor
    entries: dict[str, dict[str, Any]]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "icdor_v3",
            "source_split": "train_audit",
            "entries": self.entries,
            "sha256": self.sha256,
        }


def _edge_key(direction: str, factor_name: str, action_name: str) -> str:
    return f"{direction}:{factor_name}:{action_name}"


def build_edge_admission(
    statistics: Mapping[tuple[str, str, str], MOSAICEdgeInterventionStats],
    ontology: Mapping[str, Any],
    factor_tiers: Sequence[str],
    *,
    source_split: str,
) -> MOSAICEdgeAdmission:
    """Freeze only audit-proven Certified factor-to-action edges."""
    if source_split != "train_audit":
        raise ValueError("IC-DOR edge admission may only read train_audit interventions")
    factor_names = [factor["name"] for factor in ontology["factors"]]
    action_names = list(ontology["action_names"])
    if len(factor_tiers) != len(factor_names) or any(tier not in {"certified", "reason_only", "abstained"} for tier in factor_tiers):
        raise ValueError("IC-DOR edge admission factor tiers do not match the factor ontology")
    factor_index = ontology["factor_index"]
    action_index = ontology["action_index"]
    candidate = torch.zeros(2, len(factor_names), len(action_names), dtype=torch.bool)
    for direction_id, direction in enumerate(("support", "veto")):
        for action_name, directions in ontology["action_routes"].items():
            for edge in directions[direction]:
                candidate[direction_id, factor_index[edge["factor"]], action_index[action_name]] = True
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
                if factor_tiers[factor_id] != "certified":
                    reasons.append("factor_not_certified")
                if stats is None:
                    reasons.append("missing_train_audit_intervention")
                else:
                    if stats.valid_samples < 64:
                        reasons.append("insufficient_valid_interventions")
                    if stats.signed_effect_lcb95 <= 0:
                        reasons.append("signed_effect_lcb_not_positive")
                    if stats.tet_lcb95 <= 0:
                        reasons.append("tet_lcb_not_positive")
                    if stats.tes_lcb95 <= 0:
                        reasons.append("tes_lcb_not_positive")
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
                    "factor_tier": factor_tiers[factor_id],
                    "accepted": accepted,
                    "reasons": reasons,
                    "metrics": asdict(stats) if stats is not None else None,
                }
    unsigned = {"version": "icdor_v3", "source_split": source_split, "entries": entries}
    digest = hashlib.sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest().upper()
    return MOSAICEdgeAdmission(edge_admission_mask=admission, entries=entries, sha256=digest)
