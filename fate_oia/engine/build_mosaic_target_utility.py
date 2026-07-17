"""Build CREDO target-side utility state from independent interventions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def _unit_interval(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"target utility {field} must be numeric")
    value = float(value)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"target utility {field} must be finite")
    return max(0.0, min(1.0, value))


def _transfer_utility(row: Mapping[str, Any]) -> float:
    """Conservatively convert independently measured transfer to [0,1].

    The score requires all three directional effects to be positive. Their
    reference scales are explicit audit margins, not train/test label priors.
    ``cca`` is the fraction of correctly signed intervention responses.
    """
    if not bool(row.get("available", False)):
        return 0.0
    try:
        tes = float(row["tes"])
        tet = float(row["tet"])
        ap_delta = float(row["ap_delta"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("available target utility rows require tes, tet, and ap_delta") from error
    cca = _unit_interval(row.get("cca"), "cca")
    if min(tes, tet, ap_delta) <= 0.0:
        return 0.0
    tes_score = min(1.0, tes / 0.02)
    tet_score = min(1.0, tet / 0.02)
    ap_score = min(1.0, ap_delta / 0.01)
    return max(0.0, min(1.0, cca * tes_score * tet_score * ap_score))


def build_target_utility(transfer: Mapping[str, Any], ontology: Mapping[str, Any]) -> dict[str, Any]:
    """Create disjoint reason cS and action u matrices from ``audit_target``.

    The returned matrices retain raw audit utility. Consumers use a documented
    small learning-access floor during shadow training; final action admission
    still requires the separate edge document.
    """
    if transfer.get("source_split") != "audit_target":
        raise ValueError("CREDO target utility must be built from audit_target")
    rows = transfer.get("per_target")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("target utility requires a per_target sequence")
    factors = [str(item["name"]) for item in ontology["factors"]]
    actions = [str(item) for item in ontology["action_names"]]
    reasons = [str(item) for item in ontology["reason_names"]]
    factor_index = {name: index for index, name in enumerate(factors)}
    action_index = {name: index for index, name in enumerate(actions)}
    reason_index = {name: index for index, name in enumerate(reasons)}
    semantic = [[0.0 for _ in factors] for _ in reasons]
    action = [[0.0 for _ in actions] for _ in factors]
    row_records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("target utility rows must be mappings")
        factor_id = str(row.get("factor_id", ""))
        target_id = str(row.get("target_id", ""))
        target_name = target_id.split(":", 1)[-1]
        if factor_id not in factor_index:
            raise ValueError(f"target utility factor is not in ontology: {factor_id}")
        score = _transfer_utility(row)
        if target_name in action_index and target_id.startswith("action:"):
            action[factor_index[factor_id]][action_index[target_name]] = score
            target_type = "action"
        elif target_name in reason_index and target_id.startswith("reason:"):
            semantic[reason_index[target_name]][factor_index[factor_id]] = score
            target_type = "reason"
        else:
            raise ValueError(f"target utility target is not in ontology: {target_id}")
        row_records.append({
            "factor_id": factor_id,
            "target_id": target_id,
            "target_type": target_type,
            "available": bool(row.get("available", False)),
            "utility": score,
        })
    return {
        "source_split": "audit_target",
        "semantic_compatibility": semantic,
        "action_target_utility": action,
        "per_target_utility": row_records,
    }
