from __future__ import annotations

import re
from typing import Iterable

import torch
from torch import nn


def _tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", text.lower()) if tok]


def build_bow_text_embeddings(names: Iterable[str], dim: int = 384) -> torch.Tensor:
    """Build deterministic text prototypes from actual predicate words.

    The implementation intentionally avoids byte-hash pseudo embeddings. Each
    prototype is a normalized bag-of-words vector over the predicate ontology
    vocabulary, padded to the model dimension.
    """
    normalized = [name.replace("_", " ") for name in names]
    vocab = sorted({tok for name in normalized for tok in _tokenize(name)})
    if len(vocab) > dim:
        raise ValueError(f"Predicate vocabulary size {len(vocab)} exceeds embedding dim {dim}")
    token_to_index = {tok: i for i, tok in enumerate(vocab)}
    rows = []
    for name in normalized:
        vec = torch.zeros(dim, dtype=torch.float32)
        tokens = _tokenize(name)
        for tok in tokens:
            vec[token_to_index[tok]] += 1.0
        if tokens:
            vec /= float(len(tokens))
        rows.append(torch.nn.functional.normalize(vec, dim=0))
    return torch.stack(rows, dim=0)


def stable_text_embedding(text: str, dim: int = 384) -> torch.Tensor:
    """Backward-compatible single-text BoW embedding helper."""
    vec = torch.zeros(dim, dtype=torch.float32)
    for idx, tok in enumerate(_tokenize(text)):
        if idx >= dim:
            break
        vec[idx] += 1.0
    return torch.nn.functional.normalize(vec, dim=0)


class TextPredicateTransfer(nn.Module):
    def __init__(self, predicate_names: Iterable[str], dim: int = 384) -> None:
        super().__init__()
        names = list(predicate_names)
        proto = build_bow_text_embeddings(names, dim=dim)
        self.register_buffer("text_prototypes", proto)
        self.proj = nn.Linear(dim, dim)
        self.transfer_gate_raw = nn.Parameter(torch.zeros(len(names)))

    def forward(self, predicate_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        text = self.proj(self.text_prototypes).to(predicate_tokens.device, predicate_tokens.dtype)
        sim = torch.einsum("bpd,qd->bpq", predicate_tokens, text)
        transferred = torch.einsum("bpq,qd->bpd", torch.softmax(sim, dim=-1), text)
        gate = torch.sigmoid(self.transfer_gate_raw).to(predicate_tokens.device, predicate_tokens.dtype)
        return {
            "predicate_text_similarity": sim,
            "transferred_predicate_tokens": transferred,
            "transfer_gate": gate,
            "text_embedding_source": "ontology_bow",
        }
