from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class NativeTextAtom:
    name: str
    type: Literal["entity", "attribute", "spatial", "polarity", "action_scope"]


@dataclass(frozen=True)
class NativePredicateSpec:
    id: int
    name: str
    entity: str
    attribute: str
    spatial: str
    polarity: str
    action_scope: str
    region: str
    support_actions: list[str]
    contra_predicates: list[str]
    mirror_of: str | None = None


ATOM_FIELDS = ("entity", "attribute", "spatial", "polarity", "action_scope")


class NativeTextAtomEncoder(nn.Module):
    def __init__(self, atom_vocab: dict[str, list[str]], dim: int = 384) -> None:
        super().__init__()
        self.dim = dim
        self.vocab = {k: list(v) for k, v in atom_vocab.items()}
        self.index = {k: {name: i for i, name in enumerate(v)} for k, v in self.vocab.items()}
        self.embeddings = nn.ModuleDict({k: nn.Embedding(max(len(v), 1), dim) for k, v in self.vocab.items()})
        self.proj = nn.ModuleDict({k: nn.Linear(dim, dim, bias=False) for k in ATOM_FIELDS})
        self.type_bias = nn.Parameter(torch.zeros(dim))

    def _idx(self, field: str, value: str, device: torch.device) -> torch.Tensor:
        if value not in self.index[field]:
            raise KeyError(f"unknown atom {field}={value}")
        return torch.tensor(self.index[field][value], dtype=torch.long, device=device)

    def encode_predicates(self, predicate_specs: list[NativePredicateSpec]) -> torch.Tensor:
        device = self.type_bias.device
        rows = []
        for spec in predicate_specs:
            acc = self.type_bias
            for field in ATOM_FIELDS:
                idx = self._idx(field, getattr(spec, field), device)
                acc = acc + self.proj[field](self.embeddings[field](idx))
            rows.append(acc)
        return torch.stack(rows, dim=0)


def build_atom_vocab(specs: list[NativePredicateSpec]) -> dict[str, list[str]]:
    return {field: sorted({str(getattr(s, field)) for s in specs}) for field in ATOM_FIELDS}


def native_text_structure_loss(atom_encoder: NativeTextAtomEncoder, specs: list[NativePredicateSpec], margin: float = 0.2) -> dict[str, torch.Tensor]:
    emb = F.normalize(atom_encoder.encode_predicates(specs), dim=-1)
    by_name = {s.name: i for i, s in enumerate(specs)}
    loss = emb.sum() * 0.0
    count = 0
    for s in specs:
        for contra in s.contra_predicates:
            if contra in by_name:
                sim = (emb[s.id] * emb[by_name[contra]]).sum()
                loss = loss + F.relu(margin + sim)
                count += 1
        if s.mirror_of and s.mirror_of in by_name:
            loss = loss + (1.0 - (emb[s.id] * emb[by_name[s.mirror_of]]).sum()).pow(2)
            count += 1
    return {"native_text_structure_loss": loss / max(count, 1), "native_text_structure_pairs": torch.tensor(float(count), device=emb.device)}
