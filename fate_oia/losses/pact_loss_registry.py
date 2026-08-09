from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn


@dataclass(frozen=True)
class PACTLossTerm:
    name: str
    owner: str
    raw: Tensor
    weight: float


class PACTLossRegistry:
    def __init__(self, weights: dict[str, float]) -> None:
        self.weights = {str(k): float(v) for k, v in weights.items()}
        self.terms: dict[str, PACTLossTerm] = {}

    def add(self, name: str, owner: str, value: Tensor) -> None:
        if name in self.terms:
            raise ValueError(f"duplicate PACT loss: {name}")
        if name not in self.weights:
            raise KeyError(f"missing PACT loss weight: {name}")
        if value.ndim or not value.isfinite():
            raise ValueError(f"PACT loss {name} must be a finite scalar")
        self.terms[name] = PACTLossTerm(name, owner, value, self.weights[name])

    def total(self) -> Tensor:
        missing = set(self.weights) - set(self.terms)
        if missing:
            raise ValueError(f"uncomputed PACT losses: {sorted(missing)}")
        first = next(iter(self.terms.values())).raw
        return sum((term.weight * term.raw for term in self.terms.values()), start=first.new_zeros(()))

    def rows(self) -> list[dict]:
        return [{"name": t.name, "owner": t.owner, "raw": float(t.raw.detach()), "weight": t.weight,
                 "weighted": float((t.raw * t.weight).detach())} for t in self.terms.values()]


def exact_pact_owner_groups(model: nn.Module) -> dict[str, list[nn.Parameter]]:
    groups = {
        "shared_visual": list(model.ego.parameters()) + list(model.shared_readout.parameters()),
        "context_action": list(model.context_decoder.parameters()),
        "explanation_lane": list(model.explanation_decoder.parameters()) + list(model.predicate_reason.parameters()),
        "predicate_visual": list(model.predicate_head.parameters()),
        "action_evidence": list(model.action_evidence.parameters()) + list(model.predicate_agreement.parameters()),
        "action_contribution": list(model.action_contribution.parameters()),
        "reason_private": list(model.reason_private.parameters()),
    }
    ids = [id(p) for values in groups.values() for p in values]
    if len(ids) != len(set(ids)):
        raise ValueError("PACT optimizer ownership overlaps")
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    actual = set(ids)
    if expected != actual:
        raise ValueError(f"PACT optimizer ownership incomplete: missing={len(expected-actual)} extra={len(actual-expected)}")
    return groups
