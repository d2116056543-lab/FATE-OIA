from __future__ import annotations

import hashlib
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


def build_label_texts(ontology: dict[str, Any]) -> list[str]:
    actions = ontology.get("actions", {})
    reasons = ontology.get("reasons", {})
    action_texts = [str(actions[i] if i in actions else actions[str(i)]) for i in range(4)]
    reason_texts = [str(reasons[i] if i in reasons else reasons[str(i)]) for i in range(21)]
    if any(t.startswith("reason_") for t in reason_texts):
        raise ValueError("CAST ontology cannot contain placeholder reason_N names")
    return action_texts + reason_texts


class HashingTextPrototypeEncoder(nn.Module):
    def __init__(self, dim: int = 384, num_buckets: int = 2048):
        super().__init__()
        self.dim = int(dim)
        self.num_buckets = int(num_buckets)
        self.text_projection = nn.Linear(num_buckets, dim, bias=False)

    @staticmethod
    def _ngrams(text: str) -> list[str]:
        t = f" {text.lower().strip()} "
        grams = []
        for n in (2, 3, 4):
            grams.extend(t[i : i + n] for i in range(max(0, len(t) - n + 1)))
        grams.extend(text.lower().split())
        return grams or [text.lower()]

    def _hash_features(self, texts: list[str], device: torch.device) -> torch.Tensor:
        x = torch.zeros(len(texts), self.num_buckets, device=device)
        for row, text in enumerate(texts):
            for gram in self._ngrams(text):
                h = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16)
                x[row, h % self.num_buckets] += 1.0
        return F.normalize(x, dim=-1)

    def forward(self, label_texts: list[str]) -> torch.Tensor:
        device = self.text_projection.weight.device
        return self.text_projection(self._hash_features(label_texts, device))


class CastLabelQueryBuilder(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21):
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.text_encoder = HashingTextPrototypeEncoder(dim=dim)
        self.text_projection = self.text_encoder.text_projection
        self.learned_label_embedding = nn.Parameter(torch.zeros(action_dim + reason_dim, dim))
        self.type_embedding = nn.Embedding(2, dim)
        self.group_embedding = nn.Embedding(8, dim)
        nn.init.normal_(self.learned_label_embedding, std=0.02)

    def forward(self, label_texts: list[str]) -> dict[str, torch.Tensor]:
        proto = self.text_encoder(label_texts)
        n = proto.shape[0]
        types = torch.cat(
            [
                torch.zeros(self.action_dim, dtype=torch.long, device=proto.device),
                torch.ones(self.reason_dim, dtype=torch.long, device=proto.device),
            ],
            dim=0,
        )
        groups = torch.arange(n, device=proto.device) % self.group_embedding.num_embeddings
        queries = proto + self.learned_label_embedding[:n] + self.type_embedding(types[:n]) + self.group_embedding(groups)
        text_sim = F.normalize(proto, dim=-1) @ F.normalize(proto, dim=-1).t()
        return {
            "label_text_prototypes": proto,
            "label_queries": queries,
            "text_similarity_matrix": text_sim,
        }
