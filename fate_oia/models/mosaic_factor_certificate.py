from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _required_mapping(record: Mapping[str, Any], name: str, fields: set[str]) -> Mapping[str, Any]:
    value = record.get(name)
    if not isinstance(value, Mapping) or not fields <= set(value):
        raise ValueError(f"IC-DOR certificate record requires {name} fields {sorted(fields)}")
    return value


@dataclass(frozen=True)
class MOSAICFactorCertificate:
    """Audit-derived, immutable factor identifiability tiers.

    The certificate is deliberately a serializable audit object, never a model
    prediction.  A trainer may only build it from the disjoint train_audit
    subset and must persist the digest in checkpoints before enabling routes.
    """

    source_split: str
    entries: dict[str, dict[str, Any]]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "icdor_v3",
            "source_split": self.source_split,
            "entries": self.entries,
            "sha256": self.sha256,
        }

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tier(record: Mapping[str, Any], certified: Mapping[str, Any]) -> tuple[str, list[str]]:
    counts = _required_mapping(
        record,
        "counts",
        {"confirmed_positive", "reliable_negative", "weak_negative", "unknown", "geometry_valid"},
    )
    scores = _required_mapping(
        record,
        "scores",
        {
            "full",
            "content_only",
            "prior_only",
            "query_shuffle_drop",
            "image_shuffle_drop",
            "grounding_minus_random",
            "view_consistency",
            "mirror_consistency",
            "ece",
            "presence_variance",
            "visibility_variance",
        },
    )
    prototype = _required_mapping(record, "prototype", {"effective_count", "dominant_rate", "dead_count"})
    lcb = _required_mapping(
        record,
        "bootstrap_lcb95",
        {"full_minus_prior_only", "query_shuffle_drop", "image_shuffle_drop", "grounding_minus_random"},
    )
    count_or_geometry = (
        counts["confirmed_positive"] >= certified["min_confirmed_positive"]
        and counts["reliable_negative"] >= certified["min_reliable_negative"]
    ) or counts["geometry_valid"] >= certified["min_geometry_valid"]
    stable_content = (
        lcb["full_minus_prior_only"] > certified["min_full_minus_prior_lcb95"]
        and scores["content_only"] >= certified["min_content_fraction"] * scores["full"]
        and lcb["query_shuffle_drop"] > certified["min_query_shuffle_drop_lcb95"]
        and lcb["image_shuffle_drop"] > certified["min_image_shuffle_drop_lcb95"]
    )
    geometry_or_consistency = (
        lcb["grounding_minus_random"] > certified["min_grounding_minus_random_lcb95"]
        if counts["geometry_valid"] > 0
        else scores["view_consistency"] > 0 and scores["mirror_consistency"] > 0
    )
    prototype_ok = (
        prototype["effective_count"] > certified["min_effective_prototype_count"]
        and prototype["dominant_rate"] < certified["max_dominant_prototype_rate"]
    )
    nondegenerate = scores["presence_variance"] > 0 and scores["visibility_variance"] > 0
    if count_or_geometry and stable_content and geometry_or_consistency and prototype_ok and nondegenerate:
        return "certified", []
    if stable_content and nondegenerate:
        return "reason_only", ["insufficient_negative_grounding_or_target_utility"]
    reasons: list[str] = []
    if not stable_content:
        reasons.append("missing_content_or_shuffle_signal")
    if not prototype_ok:
        reasons.append("prototype_collapse")
    if not nondegenerate:
        reasons.append("presence_or_visibility_degenerate")
    if not geometry_or_consistency:
        reasons.append("missing_geometry_or_view_consistency")
    return "abstained", reasons or ["insufficient_identifiability"]


def _reliability_for_tier(tier: str) -> float:
    """Dataset-level route eligibility, never an image-level model prediction."""
    values = {"certified": 1.0, "reason_only": 0.5, "abstained": 0.0}
    try:
        return values[tier]
    except KeyError as error:
        raise ValueError(f"IC-DOR unknown certificate tier: {tier}") from error


def build_factor_certificate(
    factor_stats: Mapping[str, Mapping[str, Any]],
    certificate_rules: Mapping[str, Any],
    *,
    source_split: str,
) -> MOSAICFactorCertificate:
    if source_split != "train_audit":
        raise ValueError("IC-DOR factor certificates may only be built from train_audit")
    if not isinstance(certificate_rules, Mapping) or certificate_rules.get("version") != "icdor_v3":
        raise ValueError("IC-DOR factor certificate rules must be the validated icdor_v3 rules")
    certified = certificate_rules.get("certified")
    if not isinstance(certified, Mapping):
        raise ValueError("IC-DOR certificate rules require certified thresholds")
    entries: dict[str, dict[str, Any]] = {}
    for factor_name in sorted(factor_stats):
        record = factor_stats[factor_name]
        if not isinstance(record, Mapping):
            raise ValueError(f"IC-DOR factor record for {factor_name} must be a mapping")
        tier, reasons = _tier(record, certified)
        entries[factor_name] = {
            "factor": factor_name,
            "counts": dict(record["counts"]),
            "scores": dict(record["scores"]),
            "prototype": dict(record["prototype"]),
            "bootstrap_lcb95": dict(record["bootstrap_lcb95"]),
            "tier": tier,
            "reliability": _reliability_for_tier(tier),
            "reasons": reasons,
        }
    unsigned = {"version": "icdor_v3", "source_split": source_split, "entries": entries}
    digest = hashlib.sha256(_canonical_json(unsigned)).hexdigest().upper()
    return MOSAICFactorCertificate(source_split=source_split, entries=entries, sha256=digest)
