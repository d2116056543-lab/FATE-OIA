"""Build CREDO visual credibility from independent factor-audit statistics.

This conversion deliberately consumes only ``audit_visual`` bootstrap evidence.
Reason/action targets are neither accepted nor needed here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from fate_oia.models.mosaic_continuous_credibility import visual_credibility_from_measurements


_MEASUREMENT_KEYS = (
    "full_minus_prior_only",
    "query_shuffle_drop",
    "image_shuffle_drop",
    "grounding_minus_random",
    "stability",
)


def _rate(entry: Mapping[str, Any], key: str) -> float:
    value = entry.get("bootstrap_positive_rate", {}).get(key)
    if not isinstance(value, (int, float)):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _measurement(entry: Mapping[str, Any], key: str) -> tuple[float, bool]:
    """Return an audit rate plus whether that measurement was actually run."""
    values = entry.get("bootstrap_positive_rate", {})
    if not isinstance(values, Mapping):
        return 0.0, False
    value = values.get(key)
    if not isinstance(value, (int, float)):
        return 0.0, False
    return float(min(max(value, 0.0), 1.0)), True


def build_visual_credibility(
    factor_stats: Mapping[str, Mapping[str, Any]],
    *,
    factor_names: Sequence[str],
    factor_roles: Sequence[str],
    source_kinds: Sequence[str],
    image_only_cap: float = 0.10,
    unknown_cap: float = 0.0,
    no_reliable_negative_cap: float = 0.25,
) -> dict[str, Any]:
    """Return the bounded per-factor CREDO visual credibility vector.

    ``factor_stats`` must originate from the geometry-only ``audit_visual``
    collector. Missing/unavailable measurements are conservatively zero rather
    than imputed from reason annotations or model predictions.
    """
    names = tuple(str(name) for name in factor_names)
    if not names or len(names) != len(factor_roles) or len(names) != len(source_kinds):
        raise ValueError("factor names, roles, and source kinds must have one nonzero length")
    if any(name not in factor_stats for name in names):
        raise ValueError("audit_visual factor statistics are incomplete")

    values: list[float] = []
    components: dict[str, list[float]] = {key: [] for key in _MEASUREMENT_KEYS}
    availability: dict[str, list[bool]] = {
        "content": [], "query": [], "image": [], "grounding": [], "stability": [],
    }
    for name, role, source_kind in zip(names, factor_roles, source_kinds):
        entry = factor_stats[name]
        counts = entry.get("counts", {})
        positive = int(counts.get("confirmed_positive", 0))
        negative = int(counts.get("reliable_negative", 0))
        n_eff = 2.0 * positive * negative / max(positive + negative, 1)
        reliability = torch.tensor([negative > 0], dtype=torch.bool)
        content, content_available = _measurement(entry, "full_minus_prior_only")
        query, query_available = _measurement(entry, "query_shuffle_drop")
        image, image_available = _measurement(entry, "image_shuffle_drop")
        grounding, grounding_available = _measurement(entry, "grounding_minus_random")
        stability, stability_available = _measurement(entry, "stability")
        measurement = visual_credibility_from_measurements(
            content_score=torch.tensor([content]),
            prior_score=torch.tensor([0.0]),
            query_shuffle_score=torch.tensor([query]),
            image_shuffle_score=torch.tensor([image]),
            grounding_score=torch.tensor([grounding]),
            stability_score=torch.tensor([stability]),
            n_eff=torch.tensor([n_eff]),
            factor_role=str(role),
            reliable_negative=reliability,
            source_kind=str(source_kind),
            image_only_cap=image_only_cap,
            unknown_cap=unknown_cap,
            no_reliable_negative_cap=no_reliable_negative_cap,
            component_availability={
                "content": torch.tensor([content_available]),
                "query": torch.tensor([query_available]),
                "image": torch.tensor([image_available]),
                "grounding": torch.tensor([grounding_available]),
                "stability": torch.tensor([stability_available]),
            },
        )
        values.append(float(measurement["cV"].item()))
        for key, value in zip(_MEASUREMENT_KEYS, (content, query, image, grounding, stability)):
            components[key].append(value)
        for key, available in (
            ("content", content_available), ("query", query_available), ("image", image_available),
            ("grounding", grounding_available), ("stability", stability_available),
        ):
            availability[key].append(available)

    return {
        "source_split": "audit_visual",
        "reason_labels_used": False,
        "factor_names": list(names),
        "credibility": torch.tensor(values, dtype=torch.float32),
        "components": components,
        "component_availability": availability,
    }


def refresh_model_visual_credibility(model: Any, audit_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Commit one visual-only audit result to a model's detached cV EMA."""
    if audit_payload.get("source_split") != "audit_visual":
        raise ValueError("visual credibility may only be refreshed from audit_visual")
    ontology = getattr(model, "ontology", None)
    credibility_module = getattr(model, "continuous_credibility", None)
    if not isinstance(ontology, Mapping) or not callable(getattr(credibility_module, "update_from_audit", None)):
        raise ValueError("model does not expose the CREDO credibility contract")
    factors = ontology.get("factors")
    if not isinstance(factors, Sequence) or not factors:
        raise ValueError("model ontology lacks factor definitions")
    names = tuple(str(factor["name"]) for factor in factors)
    result = build_visual_credibility(
        audit_payload.get("factor_stats", {}),
        factor_names=names,
        factor_roles=tuple(str(factor.get("role", "observable")) for factor in factors),
        source_kinds=tuple(str(factor.get("source_kind", "grounded")) for factor in factors),
        image_only_cap=float(credibility_module.image_only_cap),
        unknown_cap=float(credibility_module.unknown_cap),
        no_reliable_negative_cap=float(credibility_module.no_reliable_negative_cap),
    )
    result["credibility"] = credibility_module.update_from_audit(result["credibility"])
    return result
