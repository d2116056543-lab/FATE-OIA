from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class AIECertLossTerm:
    name: str
    owner: str
    value: Tensor
    weight: float
    active: bool
    inactivity_reason: str = ""


class AIECertLossRegistry:
    def __init__(self):
        self.terms: dict[str, AIECertLossTerm] = {}

    def add(self, name: str, owner: str, value: Tensor, weight: float, active=True, reason="") -> None:
        if name in self.terms:
            raise ValueError(f"loss term {name} registered twice")
        self.terms[name] = AIECertLossTerm(name, owner, value, float(weight), bool(active), reason)

    def total(self) -> Tensor:
        if not self.terms:
            raise RuntimeError("empty AIE-CERT loss registry")
        return torch.stack([term.weight * term.value for term in self.terms.values()]).sum()


def exact_owner_parameter_groups(model) -> dict[str, list[torch.nn.Parameter]]:
    groups = {
        "primary_core": list(model.foundation.ego.parameters()) + list(model.foundation.trunk.parameters()) + list(model.foundation.predicate_reason.parameters()),
        "predicate_visual": list(model.foundation.predicate_head.parameters()),
        "action_evidence": list(model.evidence_interface.parameters()),
        "action_contribution": list(model.contribution_head.parameters()),
        "reason_private": list(model.reason_rereader.parameters()),
        "naming_readout": list(model.naming.parameters()),
    }
    ids = [id(p) for values in groups.values() for p in values if p.requires_grad]
    expected = [id(p) for p in model.parameters() if p.requires_grad and not any(id(p) == id(d) for d in model.foundation.dino.parameters())]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise RuntimeError("AIE-CERT owner groups are not an exact cover")
    return groups
