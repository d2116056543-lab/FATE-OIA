from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn
import yaml


class HashingTextPrototypeEncoder(nn.Module):
    def __init__(self, ontology_path: str | Path, dim: int = 384, action_dim: int = 4, reason_dim: int = 21) -> None:
        super().__init__()
        self.ontology_path = Path(ontology_path)
        self.dim = dim
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.num_labels = action_dim + reason_dim
        self.learned_label_embedding = nn.Parameter(torch.zeros(self.num_labels, dim))
        self.type_embedding = nn.Embedding(2, dim)
        self.group_embedding = nn.Embedding(16, dim)
        self.text_proj = nn.Linear(dim, dim)
        self.register_buffer("text_prototypes", self._build_text_prototypes(), persistent=False)
        nn.init.normal_(self.learned_label_embedding, std=0.02)

    def _load_ontology(self) -> dict[str, Any]:
        data = yaml.safe_load(self.ontology_path.read_text(encoding="utf-8"))
        actions = data.get("actions", {})
        reasons = data.get("reasons", {})
        if sorted(int(k) for k in actions.keys()) != list(range(self.action_dim)):
            raise ValueError("Ontology actions must be indexed 0..3")
        if sorted(int(k) for k in reasons.keys()) != list(range(self.reason_dim)):
            raise ValueError("Ontology reasons must be indexed 0..20")
        for idx, rec in reasons.items():
            name = str(rec.get("name", ""))
            if name.lower().startswith("reason_"):
                raise ValueError("Placeholder reason names are forbidden")
        return data

    def _hash_text(self, text: str) -> torch.Tensor:
        vec = torch.zeros(self.dim)
        for token in text.lower().replace("/", " ").replace("-", " ").split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(0, min(len(digest), 16), 2):
                idx = int.from_bytes(digest[i:i+2], "little") % self.dim
                sign = 1.0 if digest[i] % 2 == 0 else -1.0
                vec[idx] += sign
        return torch.nn.functional.normalize(vec, dim=0) if vec.norm() > 0 else vec

    def _build_text_prototypes(self) -> torch.Tensor:
        data = self._load_ontology()
        rows = []
        for i in range(self.action_dim):
            rec = data["actions"][i] if i in data["actions"] else data["actions"][str(i)]
            text = " ".join([str(rec.get("name", "")), " ".join(rec.get("aliases", [])), str(rec.get("group", "action"))])
            rows.append(self._hash_text(text))
        for i in range(self.reason_dim):
            rec = data["reasons"][i] if i in data["reasons"] else data["reasons"][str(i)]
            text = " ".join([str(rec.get("name", "")), " ".join(rec.get("aliases", [])), str(rec.get("group", "reason")), " ".join(rec.get("positive_states", []))])
            rows.append(self._hash_text(text))
        return torch.stack(rows, dim=0)

    def forward(self) -> dict[str, torch.Tensor]:
        text = self.text_proj(self.text_prototypes.to(self.learned_label_embedding.device))
        type_ids = torch.cat([torch.zeros(self.action_dim, dtype=torch.long), torch.ones(self.reason_dim, dtype=torch.long)]).to(text.device)
        group_ids = torch.arange(self.num_labels, device=text.device) % self.group_embedding.num_embeddings
        queries = text + self.learned_label_embedding + self.type_embedding(type_ids) + self.group_embedding(group_ids)
        sim = torch.nn.functional.normalize(text, dim=-1) @ torch.nn.functional.normalize(text, dim=-1).transpose(0, 1)
        return {"label_text_prototypes": text, "label_queries": queries, "text_similarity_matrix": sim}
