from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import torch
from torch import Tensor, nn


# Section 16 is deliberately represented as raw terms plus one registry weight.
# Component loss helpers must not pass their already-weighted ``total`` values.
ACTION_LOSS_TERMS = (
    "action_final",
    "action_base",
    "action_evidence_aux",
    "action_utility_cf",
    "action_utility_dense",
    "action_sufficiency",
    "action_necessity",
    "action_control",
    "action_preserve",
    "action_soft_f1",
    "action_cardinality",
    "action_easy",
)
REASON_LOSS_TERMS = (
    "reason_benchmark",
    "reason_private_direct",
    "reason_clean",
    "reason_rank",
    "reason_soft_f1",
    "reason_bbam",
    "reason_view_consistency",
    "reason_pu_private",
)
MEASUREMENT_LOSS_TERMS = (
    "measurement_anchor",
    "measurement_state",
    "measurement_null",
    "measurement_matched_background",
    "measurement_mirror",
    "measurement_identity",
)

SAVE_LOSS_WEIGHTS = {
    "action_final": 1.00,
    "action_base": 0.35,
    "action_evidence_aux": 0.20,
    "action_utility_cf": 0.10,
    "action_utility_dense": 0.02,
    "action_sufficiency": 0.08,
    "action_necessity": 0.08,
    "action_control": 0.04,
    "action_preserve": 0.02,
    "action_soft_f1": 0.03,
    "action_cardinality": 0.02,
    "action_easy": 0.03,
    "reason_benchmark": 1.00,
    "reason_private_direct": 0.35,
    "reason_clean": 0.35,
    "reason_rank": 0.06,
    "reason_soft_f1": 0.03,
    "reason_bbam": 0.03,
    "reason_view_consistency": 0.02,
    "reason_pu_private": 1.00,
    "measurement_anchor": 0.05,
    "measurement_state": 0.08,
    "measurement_null": 0.02,
    "measurement_matched_background": 0.03,
    "measurement_mirror": 0.02,
    "measurement_identity": 0.02,
}
SAVE_ACTION_LOSS_WEIGHTS = {
    name: SAVE_LOSS_WEIGHTS[name] for name in ACTION_LOSS_TERMS
}
SAVE_REASON_LOSS_WEIGHTS = {
    name: SAVE_LOSS_WEIGHTS[name] for name in REASON_LOSS_TERMS
}
SAVE_MEASUREMENT_LOSS_WEIGHTS = {
    name: SAVE_LOSS_WEIGHTS[name] for name in MEASUREMENT_LOSS_TERMS
}
SAVE_LOSS_TERM_NAMES = tuple(SAVE_LOSS_WEIGHTS)

SAVE_PARAMETER_OWNER_GROUPS = (
    "foundation_joint",
    "predicate_measurement",
    "action_multi_inquiry",
    "utility_bridge",
    "clean_reason_adapter",
    "private_reason",
)
SAVE_PARAMETER_OWNER_GROUP_SET = frozenset(SAVE_PARAMETER_OWNER_GROUPS)

# Registration ownership records which component contributes each scalar once.
# It is distinct from, but uses the same labels as, parameter gradient ownership.
SAVE_LOSS_REGISTRATION_OWNER_MAP = {
    "action_final": "action_multi_inquiry",
    "action_base": "foundation_joint",
    "action_evidence_aux": "action_multi_inquiry",
    "action_utility_cf": "utility_bridge",
    "action_utility_dense": "utility_bridge",
    "action_sufficiency": "action_multi_inquiry",
    "action_necessity": "action_multi_inquiry",
    "action_control": "action_multi_inquiry",
    "action_preserve": "action_multi_inquiry",
    "action_soft_f1": "foundation_joint",
    "action_cardinality": "foundation_joint",
    "action_easy": "foundation_joint",
    "reason_benchmark": "private_reason",
    "reason_private_direct": "private_reason",
    "reason_clean": "clean_reason_adapter",
    "reason_rank": "private_reason",
    "reason_soft_f1": "private_reason",
    "reason_bbam": "private_reason",
    "reason_view_consistency": "private_reason",
    "reason_pu_private": "private_reason",
    "measurement_anchor": "predicate_measurement",
    "measurement_state": "predicate_measurement",
    "measurement_null": "predicate_measurement",
    "measurement_matched_background": "predicate_measurement",
    "measurement_mirror": "predicate_measurement",
    "measurement_identity": "predicate_measurement",
}
SAVE_LOSS_OWNER_MAP = SAVE_LOSS_REGISTRATION_OWNER_MAP
LOSS_OWNER_MAP = SAVE_LOSS_REGISTRATION_OWNER_MAP

# Section 18's table, expressed per registered scalar. The set is an allowlist,
# not a claim that every optional branch is active on every batch.
_ACTION_FINAL_OWNERS = frozenset(
    (
        "foundation_joint",
        "predicate_measurement",
        "action_multi_inquiry",
        "utility_bridge",
        "clean_reason_adapter",
    )
)
_ACTION_MECHANISM_OWNERS = frozenset(
    ("predicate_measurement", "action_multi_inquiry", "utility_bridge")
)
_ACTION_REGULARIZER_OWNERS = frozenset(
    ("foundation_joint", "action_multi_inquiry")
)
_CLEAN_REASON_OWNERS = frozenset(
    ("foundation_joint", "predicate_measurement", "clean_reason_adapter")
)
_PRIVATE_REASON_OWNERS = frozenset(("private_reason",))

SAVE_GRADIENT_OWNER_ALLOWLIST = {
    "action_final": _ACTION_FINAL_OWNERS,
    "action_base": frozenset(("foundation_joint",)),
    "action_evidence_aux": _ACTION_MECHANISM_OWNERS,
    "action_utility_cf": _ACTION_MECHANISM_OWNERS,
    "action_utility_dense": _ACTION_MECHANISM_OWNERS,
    "action_sufficiency": _ACTION_MECHANISM_OWNERS,
    "action_necessity": _ACTION_MECHANISM_OWNERS,
    "action_control": _ACTION_MECHANISM_OWNERS,
    "action_preserve": _ACTION_MECHANISM_OWNERS,
    "action_soft_f1": _ACTION_REGULARIZER_OWNERS,
    "action_cardinality": _ACTION_REGULARIZER_OWNERS,
    "action_easy": _ACTION_REGULARIZER_OWNERS,
    "reason_benchmark": _PRIVATE_REASON_OWNERS,
    "reason_private_direct": _PRIVATE_REASON_OWNERS,
    "reason_clean": _CLEAN_REASON_OWNERS,
    "reason_rank": _PRIVATE_REASON_OWNERS,
    "reason_soft_f1": _PRIVATE_REASON_OWNERS,
    "reason_bbam": _PRIVATE_REASON_OWNERS,
    "reason_view_consistency": _PRIVATE_REASON_OWNERS,
    "reason_pu_private": _PRIVATE_REASON_OWNERS,
    "measurement_anchor": frozenset(("predicate_measurement",)),
    "measurement_state": frozenset(("predicate_measurement",)),
    "measurement_null": frozenset(("predicate_measurement",)),
    "measurement_matched_background": frozenset(("predicate_measurement",)),
    "measurement_mirror": frozenset(("predicate_measurement",)),
    "measurement_identity": frozenset(("predicate_measurement",)),
}
SAVE_LOSS_GRADIENT_OWNER_ALLOWLIST = SAVE_GRADIENT_OWNER_ALLOWLIST
GRADIENT_OWNER_ALLOWLIST = SAVE_GRADIENT_OWNER_ALLOWLIST
LOSS_WEIGHTS = SAVE_LOSS_WEIGHTS
LOSS_REGISTRATION_OWNER_MAP = SAVE_LOSS_REGISTRATION_OWNER_MAP

SAVE_PARAMETER_GROUP_LRS = {
    "foundation_joint": 8.0e-5,
    "predicate_measurement": 1.8e-4,
    "action_multi_inquiry": 1.8e-4,
    "utility_bridge": 2.0e-4,
    "clean_reason_adapter": 1.2e-4,
    "private_reason": 2.2e-4,
}
SAVE_OPTIMIZER_WEIGHT_DECAY = 0.05

# Prefixes are deliberately explicit. A model may also provide an explicit
# ``save_parameter_owner_map`` or a parameter-level ``save_parameter_owner``.
DEFAULT_SAVE_PARAMETER_OWNER_PREFIXES = {
    "foundation_joint": (
        "foundation_joint.",
        "foundation.",
        "calalign.",
        "shared_foundation.",
    ),
    "predicate_measurement": (
        "predicate_measurement.",
        "predicate.",
        "measurement.",
        "typed_evidence_state_head.",
        "typed_factors.",
    ),
    "action_multi_inquiry": (
        "action_multi_inquiry.",
        "action_inquiry.",
        "action_evidence.",
        "evidence.",
    ),
    "utility_bridge": (
        "utility_bridge.",
        "utility.",
        "action_utility.",
    ),
    "clean_reason_adapter": (
        "clean_reason_adapter.",
        "clean_reason.",
        "reason_clean.",
        "clean_reason_route.",
    ),
    "private_reason": (
        "private_reason.",
        "private_reason_decoder.",
        "reason_decoder.",
        "reason_private.",
        "pu_private.",
        "bbam.",
    ),
}

ACTION_PREDICATE_BRIDGE_SCALE = 0.05
SAVE_ACTION_PREDICATE_BRIDGE_SCALE = ACTION_PREDICATE_BRIDGE_SCALE

# A true value means that the edge is required to be detached. The mapping is
# part of the audit artifact so a trainer can report the contract without
# reverse-engineering autograd graphs from a completed run.
SAVE_FIREWALL_DETACHES = {
    "action_to_foundation_via_predicate": True,
    "grounding_to_foundation": True,
    "benchmark_to_action": True,
    "benchmark_to_clean_foundation": True,
    "pu_to_non_private": True,
    "reason_reliability_to_clean": True,
    "utility_teacher_target": True,
    "utility_dense_target": True,
    "posthoc_to_representation": True,
}
SAVE_FIREWALL_GRADIENT_ALLOWLIST = {
    "action_to_foundation_via_predicate": frozenset(
        ("predicate_measurement", "action_multi_inquiry", "utility_bridge")
    ),
    "grounding_to_foundation": frozenset(("predicate_measurement",)),
    "benchmark_to_action": frozenset(("private_reason",)),
    "benchmark_to_clean_foundation": frozenset(("private_reason",)),
    "pu_to_non_private": frozenset(("private_reason",)),
    "reason_reliability_to_clean": frozenset(
        ("foundation_joint", "predicate_measurement", "clean_reason_adapter")
    ),
    "utility_teacher_target": frozenset(
        ("predicate_measurement", "action_multi_inquiry", "utility_bridge")
    ),
    "utility_dense_target": _ACTION_MECHANISM_OWNERS,
    "posthoc_to_representation": frozenset(),
}

_POSTHOC_PARAMETER_RE = re.compile(r"(?:threshold|temperature)", re.IGNORECASE)
_DINO_PARAMETER_RE = re.compile(r"(?:^|[._])dino(?:$|[._])", re.IGNORECASE)

SAVE_LOSS_TERM_ALIASES = {
    "final": "action_final",
    "base": "action_base",
    "evidence_aux": "action_evidence_aux",
    "utility_cf": "action_utility_cf",
    "utility_dense": "action_utility_dense",
    "sufficiency": "action_sufficiency",
    "necessity": "action_necessity",
    "control": "action_control",
    "preserve": "action_preserve",
    "soft_f1": "action_soft_f1",
    "cardinality": "action_cardinality",
    "easy": "action_easy",
    "benchmark": "reason_benchmark",
    "private_direct": "reason_private_direct",
    "clean": "reason_clean",
    "rank": "reason_rank",
    "bbam": "reason_bbam",
    "view_consistency": "reason_view_consistency",
    "pu_private": "reason_pu_private",
    "anchor": "measurement_anchor",
    "state": "measurement_state",
    "null": "measurement_null",
    "observability": "measurement_null",
    "observability_null": "measurement_null",
    "measurement_null_observability": "measurement_null",
    "null_observability": "measurement_null",
    "matched_background": "measurement_matched_background",
    "mirror": "measurement_mirror",
    "identity": "measurement_identity",
}


def _canonical_loss_name(name: str) -> str:
    value = str(name)
    return SAVE_LOSS_TERM_ALIASES.get(value, value)


def _validate_static_loss_contract() -> None:
    names = set(SAVE_LOSS_TERM_NAMES)
    if set(SAVE_LOSS_REGISTRATION_OWNER_MAP) != names:
        raise RuntimeError("LOSS_OWNER_MISMATCH: registration owner map is incomplete")
    if set(SAVE_GRADIENT_OWNER_ALLOWLIST) != names:
        raise RuntimeError("OWNER_MISMATCH: gradient owner allowlist is incomplete")
    for name in SAVE_LOSS_TERM_NAMES:
        owner = SAVE_LOSS_REGISTRATION_OWNER_MAP[name]
        gradient_owners = SAVE_GRADIENT_OWNER_ALLOWLIST[name]
        if owner not in SAVE_PARAMETER_OWNER_GROUP_SET:
            raise RuntimeError("OWNER_MISMATCH: registration owner is not a valid group")
        if not gradient_owners <= SAVE_PARAMETER_OWNER_GROUP_SET:
            raise RuntimeError("OWNER_MISMATCH: gradient allowlist contains an unknown group")


_validate_static_loss_contract()


def fixed_action_predicate_bridge(value: Tensor, *, scale: float = ACTION_PREDICATE_BRIDGE_SCALE) -> Tensor:
    """Expose predicate measurements with the fixed five-percent derivative."""
    if float(scale) != ACTION_PREDICATE_BRIDGE_SCALE:
        raise ValueError("PREDICATE_ACTION_BRIDGE_SCALE is fixed at 0.05")
    return value.detach() + ACTION_PREDICATE_BRIDGE_SCALE * (value - value.detach())


selective_predicate_bridge = fixed_action_predicate_bridge


def validate_save_firewall_contract(
    detaches: Optional[Mapping[str, Any]] = None,
    *,
    bridge_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """Validate the detach and fixed-bridge contract used by SAVE branches."""
    contract = dict(SAVE_FIREWALL_DETACHES if detaches is None else detaches)
    supplied_bridge = contract.pop("predicate_action_bridge_scale", bridge_scale)
    if supplied_bridge is None:
        supplied_bridge = ACTION_PREDICATE_BRIDGE_SCALE
    if float(supplied_bridge) != ACTION_PREDICATE_BRIDGE_SCALE:
        raise ValueError("PREDICATE_ACTION_BRIDGE_SCALE must equal 0.05")
    missing = [
        name
        for name in SAVE_FIREWALL_DETACHES
        if name not in contract or not bool(contract[name])
    ]
    if missing:
        raise ValueError("FIREWALL_DETACH_MISSING: " + ", ".join(missing))
    return {
        "predicate_action_bridge_scale": ACTION_PREDICATE_BRIDGE_SCALE,
        "required_detaches": tuple(SAVE_FIREWALL_DETACHES),
        "detaches": {name: True for name in SAVE_FIREWALL_DETACHES},
        "gradient_allowlist": SAVE_FIREWALL_GRADIENT_ALLOWLIST.copy(),
    }


validate_save_firewall_detaches = validate_save_firewall_contract
validate_save_mechanism_contract = validate_save_firewall_contract


@dataclass(frozen=True)
class SAVELossTermSpec:
    name: str
    weight: float
    registration_owner: str
    gradient_owners: FrozenSet[str]

    @property
    def owner(self) -> str:
        return self.registration_owner

    @property
    def allowed_gradient_owners(self) -> FrozenSet[str]:
        return self.gradient_owners


LossTermSpec = SAVELossTermSpec


@dataclass(frozen=True)
class _SAVELossRow:
    spec: SAVELossTermSpec
    value: Tensor
    weight: float
    call_count: int = 1


class SAVELossRegistry:
    """Single-owner registry for the complete Section 16 scalar objective."""

    def __init__(
        self,
        *,
        expected_terms: Optional[Sequence[str]] = None,
        weights: Optional[Mapping[str, float]] = None,
        registration_owners: Optional[Mapping[str, str]] = None,
        gradient_owner_allowlist: Optional[Mapping[str, Iterable[str]]] = None,
    ) -> None:
        requested = SAVE_LOSS_TERM_NAMES if expected_terms is None else tuple(
            _canonical_loss_name(name) for name in expected_terms
        )
        if len(set(requested)) != len(requested):
            raise ValueError("LOSS_DUPLICATED: expected term list contains duplicates")
        unknown = set(requested) - set(SAVE_LOSS_TERM_NAMES)
        if unknown:
            raise ValueError("LOSS_UNKNOWN: " + ", ".join(sorted(unknown)))

        resolved_weights = dict(SAVE_LOSS_WEIGHTS)
        if weights is not None:
            for name, value in weights.items():
                canonical = _canonical_loss_name(name)
                if canonical not in resolved_weights:
                    raise ValueError("LOSS_UNKNOWN: " + str(name))
                resolved_weights[canonical] = float(value)
        resolved_registration = dict(SAVE_LOSS_REGISTRATION_OWNER_MAP)
        if registration_owners is not None:
            for name, value in registration_owners.items():
                canonical = _canonical_loss_name(name)
                if canonical not in resolved_registration:
                    raise ValueError("LOSS_UNKNOWN: " + str(name))
                if str(value) != SAVE_LOSS_REGISTRATION_OWNER_MAP[canonical]:
                    raise ValueError("OWNER_MISMATCH: registration owner for " + canonical)
        resolved_gradient = {
            name: frozenset(values)
            for name, values in SAVE_GRADIENT_OWNER_ALLOWLIST.items()
        }
        if gradient_owner_allowlist is not None:
            for name, values in gradient_owner_allowlist.items():
                canonical = _canonical_loss_name(name)
                if canonical not in resolved_gradient:
                    raise ValueError("LOSS_UNKNOWN: " + str(name))
                supplied = frozenset(str(value) for value in values)
                if supplied != SAVE_GRADIENT_OWNER_ALLOWLIST[canonical]:
                    raise ValueError("OWNER_MISMATCH: gradient allowlist for " + canonical)

        self._expected_terms = tuple(requested)
        self._specs = {
            name: SAVELossTermSpec(
                name=name,
                weight=float(resolved_weights[name]),
                registration_owner=resolved_registration[name],
                gradient_owners=frozenset(resolved_gradient[name]),
            )
            for name in self._expected_terms
        }
        self._rows: Dict[str, _SAVELossRow] = {}
        self._runtime_calls = {name: 0 for name in self._expected_terms}
        self._backward_calls = 0
        self._validate_specs()

    def _validate_specs(self) -> None:
        for spec in self._specs.values():
            if not math.isfinite(spec.weight) or spec.weight < 0.0:
                raise ValueError("LOSS_WEIGHT_INVALID: " + spec.name)
            if spec.registration_owner not in SAVE_PARAMETER_OWNER_GROUP_SET:
                raise ValueError("OWNER_MISMATCH: unknown registration owner for " + spec.name)
            if not spec.gradient_owners <= SAVE_PARAMETER_OWNER_GROUP_SET:
                raise ValueError("OWNER_MISMATCH: unknown gradient owner for " + spec.name)

    @property
    def expected_terms(self) -> Tuple[str, ...]:
        return self._expected_terms

    @property
    def backward_count(self) -> int:
        return self._backward_calls

    def canonical_name(self, name: str) -> str:
        canonical = _canonical_loss_name(name)
        if canonical not in self._specs:
            raise ValueError("LOSS_UNKNOWN: " + str(name))
        return canonical

    def spec(self, name: str) -> SAVELossTermSpec:
        return self._specs[self.canonical_name(name)]

    def specifications(self) -> Tuple[SAVELossTermSpec, ...]:
        return tuple(self._specs[name] for name in self._expected_terms)

    def add(
        self,
        name: str,
        value: Tensor,
        weight: Optional[float] = None,
        *,
        owner: Optional[str] = None,
        registration_owner: Optional[str] = None,
        gradient_owners: Optional[Iterable[str]] = None,
        weighted: bool = False,
        call_count: int = 1,
    ) -> None:
        canonical = self.canonical_name(name)
        if self._runtime_calls[canonical] != 0 or canonical in self._rows:
            raise ValueError("LOSS_DUPLICATED: " + canonical)
        if weighted:
            raise ValueError("LOSS_DOUBLE_WEIGHTED: register raw terms only")
        if int(call_count) != 1:
            raise ValueError("LOSS_CALL_COUNT: " + canonical)
        spec = self._specs[canonical]
        selected_weight = spec.weight if weight is None else float(weight)
        if not math.isclose(selected_weight, spec.weight, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("LOSS_DOUBLE_WEIGHTED: non-canonical weight for " + canonical)
        selected_registration_owner = registration_owner if registration_owner is not None else owner
        if selected_registration_owner is not None and str(selected_registration_owner) != spec.registration_owner:
            raise ValueError("OWNER_MISMATCH: registration owner for " + canonical)
        selected_gradient_owners = (
            spec.gradient_owners
            if gradient_owners is None
            else frozenset(str(value) for value in gradient_owners)
        )
        if selected_gradient_owners != spec.gradient_owners:
            raise ValueError("OWNER_MISMATCH: gradient allowlist for " + canonical)
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise ValueError("LOSS_SCALAR_REQUIRED: " + canonical)
        self._runtime_calls[canonical] = 1
        self._rows[canonical] = _SAVELossRow(
            spec=spec,
            value=value,
            weight=selected_weight,
            call_count=1,
        )

    register = add

    def add_weighted(self, name: str, value: Tensor, weight: Optional[float] = None, **kwargs: Any) -> None:
        del name, value, weight, kwargs
        raise ValueError("LOSS_DOUBLE_WEIGHTED: registry accepts raw terms only")

    def register_terms(self, values: Mapping[str, Tensor]) -> "SAVELossRegistry":
        for name, value in values.items():
            if str(name).lower().endswith("total"):
                raise ValueError("LOSS_DOUBLE_WEIGHTED: bundle total cannot be registered")
            self.add(name, value)
        return self

    def validate_complete(self) -> "SAVELossRegistry":
        missing = [name for name in self._expected_terms if self._runtime_calls[name] != 1]
        extra = sorted(set(self._rows) - set(self._expected_terms))
        if missing:
            raise ValueError("LOSS_MISSING: " + ", ".join(missing))
        if extra:
            raise ValueError("LOSS_UNKNOWN: " + ", ".join(extra))
        for name, row in self._rows.items():
            if row.call_count != 1 or self._runtime_calls[name] != 1:
                raise ValueError("LOSS_CALL_COUNT: " + name)
            if not math.isclose(row.weight, row.spec.weight, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("LOSS_DOUBLE_WEIGHTED: " + name)
        return self

    def total(self, *, require_complete: bool = True) -> Tensor:
        if require_complete:
            self.validate_complete()
        if not self._rows:
            return torch.zeros(())
        first = next(iter(self._rows.values())).value
        total = first.new_zeros(())
        for row in self._rows.values():
            total = total + row.weight * row.value
        return total

    def backward(self, *args: Any, **kwargs: Any) -> Tensor:
        if self._backward_calls != 0:
            raise ValueError("BACKWARD_DUPLICATED: SAVE objective must backward once")
        self._backward_calls = 1
        total = self.total()
        total.backward(*args, **kwargs)
        return total

    def owner_total(self, owners: Iterable[str] | str) -> Tensor:
        selected = {str(owners)} if isinstance(owners, str) else {str(value) for value in owners}
        rows = [row for row in self._rows.values() if row.spec.registration_owner in selected]
        if not rows:
            return self.total(require_complete=False).new_zeros(())
        total = rows[0].value.new_zeros(())
        for row in rows:
            total = total + row.weight * row.value
        return total

    def raw_values(self) -> Dict[str, Tensor]:
        return {name: row.value for name, row in self._rows.items()}

    def weighted_values(self) -> Dict[str, Tensor]:
        return {name: row.weight * row.value for name, row in self._rows.items()}

    def runtime_call_counts(self) -> Dict[str, int]:
        return dict(self._runtime_calls)

    def call_count(self, name: str) -> int:
        return int(self._runtime_calls[self.canonical_name(name)])

    @property
    def call_counts(self) -> Dict[str, int]:
        return self.runtime_call_counts()

    def loss_owner_map(self) -> Dict[str, str]:
        return {
            name: spec.registration_owner
            for name, spec in self._specs.items()
        }

    def gradient_owner_allowlist(self) -> Dict[str, FrozenSet[str]]:
        return {name: spec.gradient_owners for name, spec in self._specs.items()}

    def artifact(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for name in self._expected_terms:
            spec = self._specs[name]
            row = self._rows.get(name)
            value = None if row is None else float(row.value.detach().cpu())
            weighted_value = None if row is None else float((row.weight * row.value).detach().cpu())
            rows.append(
                {
                    "term": name,
                    "owner": spec.registration_owner,
                    "registration_owner": spec.registration_owner,
                    "gradient_owners": tuple(sorted(spec.gradient_owners)),
                    "weight": spec.weight,
                    "call_count": 0 if row is None else row.call_count,
                    "value": value,
                    "weighted_value": weighted_value,
                }
            )
        return rows

    def audit(self) -> Dict[str, Any]:
        return {
            "runtime_call_counts": self.runtime_call_counts(),
            "loss_owner_map": self.loss_owner_map(),
            "gradient_owner_allowlist": {
                name: tuple(sorted(values))
                for name, values in self.gradient_owner_allowlist().items()
            },
            "terms": self.artifact(),
            "backward_count": self._backward_calls,
        }

    def runtime(self) -> "SAVELossRuntime":
        return SAVELossRuntime(self)


SAVE_LOSS_TERM_SPECS = {
    name: SAVELossTermSpec(
        name=name,
        weight=float(SAVE_LOSS_WEIGHTS[name]),
        registration_owner=SAVE_LOSS_REGISTRATION_OWNER_MAP[name],
        gradient_owners=frozenset(SAVE_GRADIENT_OWNER_ALLOWLIST[name]),
    )
    for name in SAVE_LOSS_TERM_NAMES
}
LOSS_TERM_SPECS = SAVE_LOSS_TERM_SPECS


class SAVELossRuntime:
    """Trainer facade that makes one invocation and one registration atomic."""

    def __init__(self, registry: Optional[SAVELossRegistry] = None) -> None:
        self.registry = SAVELossRegistry() if registry is None else registry

    def invoke(
        self,
        name: str,
        function: Callable[..., Tensor],
        *args: Any,
        weighted: bool = False,
        **kwargs: Any,
    ) -> Tensor:
        canonical = self.registry.canonical_name(name)
        if self.registry.runtime_call_counts()[canonical] != 0:
            raise ValueError("LOSS_DUPLICATED: " + canonical)
        value = function(*args, **kwargs)
        self.registry.add(canonical, value, weighted=bool(weighted))
        return value

    call = invoke

    def add(self, *args: Any, **kwargs: Any) -> None:
        self.registry.add(*args, **kwargs)

    def validate_complete(self) -> SAVELossRegistry:
        return self.registry.validate_complete()

    def total(self, **kwargs: Any) -> Tensor:
        return self.registry.total(**kwargs)

    def backward(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.registry.backward(*args, **kwargs)

    def runtime_call_counts(self) -> Dict[str, int]:
        return self.registry.runtime_call_counts()

    def artifact(self) -> List[Dict[str, Any]]:
        return self.registry.artifact()


_ACTION_BUNDLE_KEYS = {
    "final": "action_final",
    "base": "action_base",
    "evidence_aux": "action_evidence_aux",
    "utility_cf": "action_utility_cf",
    "utility_dense": "action_utility_dense",
    "sufficiency": "action_sufficiency",
    "necessity": "action_necessity",
    "control": "action_control",
    "preserve": "action_preserve",
    "soft_f1": "action_soft_f1",
    "cardinality": "action_cardinality",
    "easy": "action_easy",
}
_REASON_BUNDLE_KEYS = {
    "benchmark": "reason_benchmark",
    "private_direct": "reason_private_direct",
    "clean": "reason_clean",
    "rank": "reason_rank",
    "soft_f1": "reason_soft_f1",
    "bbam": "reason_bbam",
    "view_consistency": "reason_view_consistency",
    "pu_private": "reason_pu_private",
}
_MEASUREMENT_BUNDLE_KEYS = {
    "anchor": "measurement_anchor",
    "state": "measurement_state",
    "null": "measurement_null",
    "observability": "measurement_null",
    "matched_background": "measurement_matched_background",
    "mirror": "measurement_mirror",
    "identity": "measurement_identity",
}


def register_save_loss_bundles(
    registry: SAVELossRegistry,
    *,
    action: Mapping[str, Tensor],
    reason: Mapping[str, Tensor],
    measurement: Mapping[str, Tensor],
) -> SAVELossRegistry:
    """Register raw category terms and reject pre-aggregated bundle totals."""
    for bundle, keys in (
        (action, _ACTION_BUNDLE_KEYS),
        (reason, _REASON_BUNDLE_KEYS),
        (measurement, _MEASUREMENT_BUNDLE_KEYS),
    ):
        for local_name, value in bundle.items():
            if str(local_name).lower().endswith("total"):
                raise ValueError("LOSS_DOUBLE_WEIGHTED: bundle totals are not registry terms")
            canonical = keys.get(str(local_name), _canonical_loss_name(local_name))
            if canonical not in keys.values():
                raise ValueError("LOSS_UNKNOWN: " + str(local_name))
            registry.add(canonical, value)
    registry.validate_complete()
    return registry


def build_save_loss_registry(
    *,
    action: Mapping[str, Tensor],
    reason: Mapping[str, Tensor],
    measurement: Mapping[str, Tensor],
) -> SAVELossRegistry:
    registry = SAVELossRegistry()
    return register_save_loss_bundles(
        registry,
        action=action,
        reason=reason,
        measurement=measurement,
    )


def validate_loss_gradient_owners(name: str, observed_owners: Iterable[str]) -> FrozenSet[str]:
    canonical = _canonical_loss_name(name)
    if canonical not in SAVE_GRADIENT_OWNER_ALLOWLIST:
        raise ValueError("LOSS_UNKNOWN: " + str(name))
    observed = frozenset(str(value) for value in observed_owners)
    allowed = SAVE_GRADIENT_OWNER_ALLOWLIST[canonical]
    if not observed <= allowed:
        unexpected = sorted(observed - allowed)
        raise ValueError("OWNER_MISMATCH: " + canonical + " reached " + ", ".join(unexpected))
    return allowed


assert_gradient_owner_allowlist = validate_loss_gradient_owners


@dataclass(frozen=True)
class SAVEParameterOwnershipReport:
    owner_sets: Dict[str, FrozenSet[str]]
    parameter_owners: Dict[str, str]
    ignored_parameter_names: Tuple[str, ...] = ()
    dino_parameter_names: Tuple[str, ...] = ()
    posthoc_parameter_names: Tuple[str, ...] = ()

    @property
    def owner_by_name(self) -> Dict[str, str]:
        return dict(self.parameter_owners)

    @property
    def parameter_owner_map(self) -> Dict[str, str]:
        return dict(self.parameter_owners)


OptimizerOwnershipReport = SAVEParameterOwnershipReport


def _named_parameter_items(source: Any) -> List[Tuple[str, Tensor]]:
    if isinstance(source, nn.Module):
        return [(str(name), parameter) for name, parameter in source.named_parameters()]
    if isinstance(source, Mapping):
        return [(str(name), parameter) for name, parameter in source.items()]
    try:
        return [(str(name), parameter) for name, parameter in source]
    except (TypeError, ValueError) as exc:
        raise TypeError("parameter source must be a module, mapping, or named-parameter iterable") from exc


def _is_posthoc_parameter(name: str) -> bool:
    return _POSTHOC_PARAMETER_RE.search(str(name)) is not None


def _is_dino_parameter(name: str) -> bool:
    return _DINO_PARAMETER_RE.search(str(name)) is not None


def _parameter_owner_from_model(parameter: Tensor) -> Optional[str]:
    for attribute in ("save_parameter_owner", "_save_parameter_owner", "parameter_owner"):
        value = getattr(parameter, attribute, None)
        if value is not None:
            return str(value)
    return None


def _resolve_parameter_owner(
    name: str,
    parameter: Tensor,
    *,
    owner_prefixes: Mapping[str, Sequence[str]],
    explicit_owners: Optional[Mapping[Any, str]],
) -> str:
    explicit: Optional[str] = None
    if explicit_owners is not None:
        if name in explicit_owners:
            explicit = str(explicit_owners[name])
        elif id(parameter) in explicit_owners:
            explicit = str(explicit_owners[id(parameter)])
    if explicit is None:
        explicit = _parameter_owner_from_model(parameter)
    if explicit is not None:
        if explicit not in SAVE_PARAMETER_OWNER_GROUP_SET:
            raise ValueError("OWNER_MISMATCH: unknown parameter owner " + explicit)
        return explicit

    matches = [
        group
        for group, prefixes in owner_prefixes.items()
        if group in SAVE_PARAMETER_OWNER_GROUP_SET
        and any(
            str(name).startswith(str(prefix))
            for prefix in ((prefixes,) if isinstance(prefixes, str) else prefixes)
        )
    ]
    if len(matches) > 1:
        raise ValueError("DUPLICATE_OWNER: " + str(name) + " matches " + ", ".join(sorted(matches)))
    if not matches:
        raise ValueError("UNOWNED_PARAMETER: " + str(name))
    return matches[0]


def validate_save_parameter_ownership(
    parameters: Any,
    *,
    owner_prefixes: Optional[Mapping[str, Sequence[str]]] = None,
    parameter_owner_map: Optional[Mapping[Any, str]] = None,
    reject_posthoc: bool = True,
) -> SAVEParameterOwnershipReport:
    """Return six disjoint owner sets covering trainable non-DINO parameters."""
    pairs = _named_parameter_items(parameters)
    prefixes = DEFAULT_SAVE_PARAMETER_OWNER_PREFIXES if owner_prefixes is None else owner_prefixes
    if parameter_owner_map is None and isinstance(parameters, nn.Module):
        model_owner_map = getattr(parameters, "save_parameter_owner_map", None)
        if isinstance(model_owner_map, Mapping):
            parameter_owner_map = model_owner_map
    owner_sets: Dict[str, set[str]] = {group: set() for group in SAVE_PARAMETER_OWNER_GROUPS}
    parameter_owners: Dict[str, str] = {}
    parameter_ids: Dict[int, str] = {}
    ignored: List[str] = []
    dino: List[str] = []
    posthoc: List[str] = []

    for name, parameter in pairs:
        if not isinstance(parameter, Tensor):
            raise TypeError("parameter " + name + " is not a tensor")
        if id(parameter) in parameter_ids:
            raise ValueError("DUPLICATE_OWNER: parameter object appears as " + parameter_ids[id(parameter)] + " and " + name)
        parameter_ids[id(parameter)] = name
        if _is_posthoc_parameter(name):
            posthoc.append(name)
            if bool(parameter.requires_grad) and reject_posthoc:
                raise ValueError("POSTHOC_PARAMETER_INCLUDED: " + name)
            ignored.append(name)
            continue
        if _is_dino_parameter(name):
            dino.append(name)
            if bool(parameter.requires_grad):
                raise ValueError("DINO_TRAINABLE: " + name)
            ignored.append(name)
            continue
        if not bool(parameter.requires_grad):
            ignored.append(name)
            continue
        owner = _resolve_parameter_owner(
            name,
            parameter,
            owner_prefixes=prefixes,
            explicit_owners=parameter_owner_map,
        )
        if name in parameter_owners:
            raise ValueError("DUPLICATE_OWNER: " + name)
        parameter_owners[name] = owner
        owner_sets[owner].add(name)

    expected = set(parameter_owners)
    actual = set().union(*owner_sets.values()) if owner_sets else set()
    if expected != actual:
        raise ValueError("UNOWNED_PARAMETER: parameter owner sets do not cover the trainable set")
    if len(actual) != sum(len(values) for values in owner_sets.values()):
        raise ValueError("DUPLICATE_OWNER: parameter owner sets overlap")
    return SAVEParameterOwnershipReport(
        owner_sets={group: frozenset(owner_sets[group]) for group in SAVE_PARAMETER_OWNER_GROUPS},
        parameter_owners=dict(parameter_owners),
        ignored_parameter_names=tuple(ignored),
        dino_parameter_names=tuple(dino),
        posthoc_parameter_names=tuple(posthoc),
    )


def save_parameter_owner_sets(parameters: Any, **kwargs: Any) -> Dict[str, FrozenSet[str]]:
    return validate_save_parameter_ownership(parameters, **kwargs).owner_sets


parameter_owner_sets = save_parameter_owner_sets


def build_save_optimizer_groups(
    model: Any,
    *,
    learning_rates: Optional[Mapping[str, float]] = None,
    weight_decay: float = SAVE_OPTIMIZER_WEIGHT_DECAY,
    owner_prefixes: Optional[Mapping[str, Sequence[str]]] = None,
    parameter_owner_map: Optional[Mapping[Any, str]] = None,
) -> List[Dict[str, Any]]:
    """Build the six AdamW groups after exact-cover validation."""
    report = validate_save_parameter_ownership(
        model,
        owner_prefixes=owner_prefixes,
        parameter_owner_map=parameter_owner_map,
    )
    by_name = {name: parameter for name, parameter in _named_parameter_items(model)}
    rates = dict(SAVE_PARAMETER_GROUP_LRS)
    if learning_rates is not None:
        unknown = set(learning_rates) - SAVE_PARAMETER_OWNER_GROUP_SET
        if unknown:
            raise ValueError("OWNER_MISMATCH: unknown optimizer group " + ", ".join(sorted(unknown)))
        rates.update({name: float(value) for name, value in learning_rates.items()})
    groups: List[Dict[str, Any]] = []
    for group_name in SAVE_PARAMETER_OWNER_GROUPS:
        names = sorted(report.owner_sets[group_name])
        groups.append(
            {
                "params": [by_name[name] for name in names],
                "lr": float(rates[group_name]),
                "weight_decay": float(weight_decay),
                "group_name": group_name,
                "name": group_name,
            }
        )
    return groups


def validate_optimizer_groups(
    groups_or_optimizer: Any,
    model_or_parameters: Any,
    *,
    owner_prefixes: Optional[Mapping[str, Sequence[str]]] = None,
    parameter_owner_map: Optional[Mapping[Any, str]] = None,
) -> SAVEParameterOwnershipReport:
    """Validate optimizer groups against the same exact-cover report."""
    report = validate_save_parameter_ownership(
        model_or_parameters,
        owner_prefixes=owner_prefixes,
        parameter_owner_map=parameter_owner_map,
    )
    groups = getattr(groups_or_optimizer, "param_groups", groups_or_optimizer)
    try:
        group_list = list(groups)
    except TypeError as exc:
        raise TypeError("optimizer groups must be a sequence or optimizer") from exc
    by_id = {
        id(parameter): name
        for name, parameter in _named_parameter_items(model_or_parameters)
    }
    expected_ids = {id(parameter) for name, parameter in _named_parameter_items(model_or_parameters) if name in report.parameter_owners}
    seen_ids: set[int] = set()
    seen_groups: set[str] = set()
    for group in group_list:
        group_name = group.get("group_name", group.get("name"))
        if group_name not in SAVE_PARAMETER_OWNER_GROUP_SET:
            raise ValueError("OWNER_MISMATCH: unknown optimizer group " + str(group_name))
        if group_name in seen_groups:
            raise ValueError("DUPLICATE_OWNER: optimizer group " + group_name)
        seen_groups.add(group_name)
        for parameter in group.get("params", ()):
            parameter_id = id(parameter)
            if parameter_id in seen_ids:
                raise ValueError("DUPLICATE_OWNER: optimizer parameter appears in multiple groups")
            seen_ids.add(parameter_id)
            if parameter_id not in by_id:
                raise ValueError("UNOWNED_PARAMETER: optimizer contains an unknown parameter")
            name = by_id[parameter_id]
            if _is_posthoc_parameter(name):
                raise ValueError("POSTHOC_PARAMETER_INCLUDED: " + name)
            if name not in report.parameter_owners:
                raise ValueError("UNOWNED_PARAMETER: " + name)
            if report.parameter_owners[name] != group_name:
                raise ValueError("OWNER_MISMATCH: " + name + " is in " + group_name)
    missing_ids = expected_ids - seen_ids
    if missing_ids:
        missing_names = sorted(by_id[parameter_id] for parameter_id in missing_ids)
        raise ValueError("UNOWNED_PARAMETER: optimizer is missing " + ", ".join(missing_names))
    if seen_groups != SAVE_PARAMETER_OWNER_GROUP_SET:
        missing_groups = sorted(SAVE_PARAMETER_OWNER_GROUP_SET - seen_groups)
        raise ValueError("UNOWNED_PARAMETER: optimizer groups missing " + ", ".join(missing_groups))
    return report


validate_save_optimizer_groups = validate_optimizer_groups
validate_optimizer_owner_exact_cover = validate_optimizer_groups
build_optimizer_param_groups = build_save_optimizer_groups


def build_save_optimizer(
    model: Any,
    *,
    optimizer_cls: Any = torch.optim.AdamW,
    learning_rates: Optional[Mapping[str, float]] = None,
    weight_decay: float = SAVE_OPTIMIZER_WEIGHT_DECAY,
    **optimizer_kwargs: Any,
) -> Any:
    groups = build_save_optimizer_groups(
        model,
        learning_rates=learning_rates,
        weight_decay=weight_decay,
    )
    optimizer = optimizer_cls(groups, **optimizer_kwargs)
    validate_optimizer_groups(optimizer, model)
    return optimizer


def optimizer_for_save(model: Any, **kwargs: Any) -> Any:
    return build_save_optimizer(model, **kwargs)


__all__ = [
    "ACTION_LOSS_TERMS",
    "ACTION_PREDICATE_BRIDGE_SCALE",
    "DEFAULT_SAVE_PARAMETER_OWNER_PREFIXES",
    "GRADIENT_OWNER_ALLOWLIST",
    "LOSS_OWNER_MAP",
    "LOSS_REGISTRATION_OWNER_MAP",
    "LOSS_TERM_SPECS",
    "LOSS_WEIGHTS",
    "LossTermSpec",
    "MEASUREMENT_LOSS_TERMS",
    "OptimizerOwnershipReport",
    "REASON_LOSS_TERMS",
    "SAVE_ACTION_LOSS_WEIGHTS",
    "SAVE_ACTION_PREDICATE_BRIDGE_SCALE",
    "SAVE_FIREWALL_DETACHES",
    "SAVE_FIREWALL_GRADIENT_ALLOWLIST",
    "SAVE_GRADIENT_OWNER_ALLOWLIST",
    "SAVE_LOSS_GRADIENT_OWNER_ALLOWLIST",
    "SAVE_LOSS_OWNER_MAP",
    "SAVE_LOSS_REGISTRATION_OWNER_MAP",
    "SAVE_LOSS_TERM_ALIASES",
    "SAVE_LOSS_TERM_NAMES",
    "SAVE_LOSS_TERM_SPECS",
    "SAVE_LOSS_WEIGHTS",
    "SAVE_MEASUREMENT_LOSS_WEIGHTS",
    "SAVE_OPTIMIZER_WEIGHT_DECAY",
    "SAVE_PARAMETER_GROUP_LRS",
    "SAVE_PARAMETER_OWNER_GROUPS",
    "SAVE_PARAMETER_OWNER_GROUP_SET",
    "SAVE_REASON_LOSS_WEIGHTS",
    "SAVEParameterOwnershipReport",
    "SAVELossRegistry",
    "SAVELossRuntime",
    "SAVELossTermSpec",
    "assert_gradient_owner_allowlist",
    "build_optimizer_param_groups",
    "build_save_loss_registry",
    "build_save_optimizer",
    "build_save_optimizer_groups",
    "fixed_action_predicate_bridge",
    "optimizer_for_save",
    "parameter_owner_sets",
    "register_save_loss_bundles",
    "save_parameter_owner_sets",
    "selective_predicate_bridge",
    "validate_loss_gradient_owners",
    "validate_optimizer_groups",
    "validate_optimizer_owner_exact_cover",
    "validate_save_firewall_contract",
    "validate_save_firewall_detaches",
    "validate_save_mechanism_contract",
    "validate_save_optimizer_groups",
    "validate_save_parameter_ownership",
]
