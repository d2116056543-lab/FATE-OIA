"""Strict, typed semantic schema support for RAEL-OIA.

The ontology is intentionally limited to compositional semantic priors.  It
does not expose action compatibility, labels, or any supervision-time state.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml
from yaml.constructor import ConstructorError


ROLE_NAMES = frozenset(
    {
        "forward_support",
        "forward_veto",
        "stop_support",
        "stop_veto",
        "left_support",
        "left_veto",
        "right_support",
        "right_veto",
        "neutral_context",
    }
)
_EXACT_ROW_FIELDS = frozenset(
    {
        "id",
        "name",
        "entity",
        "state",
        "sector",
        "role",
        "mirror_partner",
        "explicit_evidence_families",
        "pu_eligible",
    }
)
_EXACT_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "role_application",
        "hard_action_masks",
        "reasons",
    }
)
_PLACEHOLDER = re.compile(r"^(reason[_ -]?\d+|unknown|tbd|placeholder)$", re.IGNORECASE)
_MIRRORED_ROLES = {
    "left_support": "right_support",
    "right_support": "left_support",
    "left_veto": "right_veto",
    "right_veto": "left_veto",
}
_MIRRORED_SECTORS = {
    "left_corridor": "right_corridor",
    "right_corridor": "left_corridor",
    "upper_control": "upper_control",
    "front_center": "front_center",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys at every level."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key ({key!r})",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True)
class ReasonSemanticRow:
    """One immutable compositional semantic row from the 21-reason ontology."""

    id: int
    name: str
    entity: str
    state: str
    sector: str
    role: str
    mirror_partner: int | None
    explicit_evidence_families: tuple[str, ...]
    pu_eligible: bool


def _require_nonempty_string(value: object, field: str, row_id: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"reason {row_id!r} has invalid {field}")
    return value.strip()


def _validate_row(raw: Mapping[str, object]) -> ReasonSemanticRow:
    row_fields = frozenset(raw)
    missing = sorted(_EXACT_ROW_FIELDS.difference(row_fields))
    extra = sorted((str(field) for field in row_fields.difference(_EXACT_ROW_FIELDS)))
    row_id = raw.get("id")
    if missing:
        raise ValueError(
            f"reason {row_id!r} is missing required fields {missing}; "
            "each reason must declare exactly the nine schema fields"
        )
    if extra:
        raise ValueError(
            f"reason {row_id!r} must declare exactly the nine schema fields; "
            f"unexpected={extra}"
        )
    if not isinstance(row_id, int) or isinstance(row_id, bool):
        raise ValueError(f"reason id must be an integer, got {row_id!r}")

    name = _require_nonempty_string(raw["name"], "name", row_id)
    if _PLACEHOLDER.fullmatch(name):
        raise ValueError(f"reason {row_id} uses placeholder name {name!r}")
    entity = _require_nonempty_string(raw["entity"], "entity", row_id)
    state = _require_nonempty_string(raw["state"], "state", row_id)
    sector = _require_nonempty_string(raw["sector"], "sector", row_id)
    role = _require_nonempty_string(raw["role"], "role", row_id)
    if role not in ROLE_NAMES:
        raise ValueError(f"reason {row_id} has unsupported semantic role {role!r}")

    mirror_partner = raw["mirror_partner"]
    if mirror_partner is not None and (
        not isinstance(mirror_partner, int) or isinstance(mirror_partner, bool)
    ):
        raise ValueError(f"reason {row_id} has invalid mirror partner {mirror_partner!r}")
    families = raw["explicit_evidence_families"]
    if not isinstance(families, list) or not families:
        raise ValueError(f"reason {row_id} must declare nonempty explicit evidence families")
    normalized_families = tuple(_require_nonempty_string(value, "explicit evidence family", row_id) for value in families)
    if len(set(normalized_families)) != len(normalized_families):
        raise ValueError(f"reason {row_id} repeats an explicit evidence family")
    if not isinstance(raw["pu_eligible"], bool):
        raise ValueError(f"reason {row_id} has nonboolean pu_eligible")

    return ReasonSemanticRow(
        id=row_id,
        name=name,
        entity=entity,
        state=state,
        sector=sector,
        role=role,
        mirror_partner=mirror_partner,
        explicit_evidence_families=normalized_families,
        pu_eligible=raw["pu_eligible"],
    )


_DIRECTION_TOKEN = re.compile(r"(?<![A-Za-z])(?P<side>left|right)(?![A-Za-z])", re.IGNORECASE)


def _swap_directional_token(match: re.Match[str]) -> str:
    source = match.group("side")
    target = "right" if source.lower() == "left" else "left"
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target.capitalize()
    return target


def _directional_token_mirror(value: str) -> str:
    """Mirror complete left/right tokens across underscore, space, and hyphen boundaries."""

    return _DIRECTION_TOKEN.sub(_swap_directional_token, value)


def _matches_directional_mirror(source: str, partner: str) -> bool:
    mirrored_source = _directional_token_mirror(source)
    mirrored_partner = _directional_token_mirror(partner)
    source_is_directional = mirrored_source != source
    partner_is_directional = mirrored_partner != partner
    if source_is_directional or partner_is_directional:
        return (
            source_is_directional
            and partner_is_directional
            and partner == mirrored_source
            and source == mirrored_partner
        )
    return source == partner


def _validate_mirrors(rows: tuple[ReasonSemanticRow, ...]) -> None:
    by_id = {row.id: row for row in rows}
    for row in rows:
        partner = row.mirror_partner
        if partner is None:
            continue
        if partner not in by_id or partner == row.id:
            raise ValueError(f"reason {row.id} has invalid mirror partner {partner!r}")
        counterpart = by_id[partner]
        if counterpart.mirror_partner != row.id:
            raise ValueError(f"reason mirror relation {row.id}<->{partner} is not symmetric")
        expected_role = _MIRRORED_ROLES.get(row.role)
        if expected_role is None:
            raise ValueError(
                f"reason mirror pair {row.id}<->{partner} must use a left/right role pair, "
                f"got {row.role!r}"
            )
        if counterpart.role != expected_role:
            raise ValueError(
                f"reason mirror pair {row.id}<->{partner} has incompatible role "
                f"{row.role!r}->{counterpart.role!r}"
            )
        expected_sector = _MIRRORED_SECTORS.get(row.sector)
        if expected_sector is None or counterpart.sector != expected_sector:
            raise ValueError(
                f"reason mirror pair {row.id}<->{partner} has incompatible sector "
                f"{row.sector!r}->{counterpart.sector!r}"
            )
        if not _matches_directional_mirror(row.name, counterpart.name):
            raise ValueError(f"reason mirror pair {row.id}<->{partner} has incompatible name")
        if not _matches_directional_mirror(row.entity, counterpart.entity):
            raise ValueError(f"reason mirror pair {row.id}<->{partner} has incompatible entity")
        if not _matches_directional_mirror(row.state, counterpart.state):
            raise ValueError(f"reason mirror pair {row.id}<->{partner} has incompatible state")
        if len(row.explicit_evidence_families) != len(counterpart.explicit_evidence_families):
            raise ValueError(
                f"reason mirror pair {row.id}<->{partner} has incompatible explicit evidence families"
            )
        if any(
            not _matches_directional_mirror(source, target)
            for source, target in zip(row.explicit_evidence_families, counterpart.explicit_evidence_families)
        ):
            raise ValueError(
                f"reason mirror pair {row.id}<->{partner} has incompatible explicit evidence families"
            )


def load_reason_semantic_schema(path: str | Path) -> tuple[ReasonSemanticRow, ...]:
    """Load the exact RAEL 21-row semantic ontology without silent coercions."""

    schema_path = Path(path)
    if not schema_path.is_file():
        raise FileNotFoundError(f"RAEL reason schema is missing: {schema_path}")
    payload = yaml.load(schema_path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    if not isinstance(payload, Mapping):
        raise ValueError("RAEL reason schema must be a mapping")
    payload_fields = frozenset(payload)
    missing_top_level = sorted(_EXACT_TOP_LEVEL_FIELDS.difference(payload_fields))
    unexpected_top_level = sorted(str(value) for value in payload_fields.difference(_EXACT_TOP_LEVEL_FIELDS))
    if missing_top_level or unexpected_top_level:
        raise ValueError(
            "RAEL reason schema must declare exactly the four top-level fields; "
            f"missing={missing_top_level}, unexpected={unexpected_top_level}"
        )
    if payload.get("schema_version") != 1:
        raise ValueError("RAEL reason schema must use schema_version=1")
    if payload.get("role_application") != "soft_prior_only" or payload.get("hard_action_masks") is not False:
        raise ValueError("RAEL roles must remain soft priors; hard action masks are forbidden")
    raw_rows = payload.get("reasons")
    if not isinstance(raw_rows, list) or len(raw_rows) != 21:
        raise ValueError("RAEL reason schema must contain exactly 21 rows")
    if not all(isinstance(row, Mapping) for row in raw_rows):
        raise ValueError("every RAEL reason row must be a mapping")
    rows = tuple(_validate_row(row) for row in raw_rows)
    ids = tuple(row.id for row in rows)
    if set(ids) != set(range(21)) or len(set(ids)) != 21:
        raise ValueError("RAEL reason ids must be the exact unique range 0..20")
    rows = tuple(sorted(rows, key=lambda row: row.id))
    _validate_mirrors(rows)
    return rows


__all__ = ["ROLE_NAMES", "ReasonSemanticRow", "load_reason_semantic_schema"]
