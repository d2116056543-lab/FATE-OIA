from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
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


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_dict_from_checkpoint(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return {str(k): v for k, v in value.items() if torch.is_tensor(v)}
        return {str(k): v for k, v in checkpoint.items() if torch.is_tensor(v)}
    return {}


def load_oia_predicate_queries(
    checkpoint_path: str | Path,
    *,
    expected_count: int = 32,
    preferred_key: str = "predicate_head.predicate_queries",
) -> tuple[torch.Tensor, dict[str, object]]:
    """Load the learned ACPR-CalAlign OIA predicate query/prototype tensor.

    The source plan requires this to be an actual checkpoint tensor, not a
    fabricated text prior.  Ambiguity is treated as an error because otherwise
    the PSI transfer branch could silently train without the intended OIA prior.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"OIA ACPR checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    state = _state_dict_from_checkpoint(checkpoint)
    candidates: list[tuple[str, torch.Tensor]] = []
    if preferred_key in state and state[preferred_key].ndim == 2 and state[preferred_key].shape[0] == expected_count:
        candidates.append((preferred_key, state[preferred_key].detach().float()))
    else:
        for key, value in state.items():
            lower = key.lower()
            if not any(token in lower for token in ("predicate", "query", "proto")):
                continue
            if value.ndim == 2 and value.shape[0] == expected_count:
                candidates.append((key, value.detach().float()))
    unique = []
    seen = set()
    for key, value in candidates:
        if key not in seen:
            unique.append((key, value))
            seen.add(key)
    if len(unique) != 1:
        keys = [key for key, _ in unique]
        raise ValueError(f"Ambiguous OIA predicate query tensor resolution: {keys}")
    source_key, tensor = unique[0]
    report = {
        "source_checkpoint_path": str(path),
        "source_checkpoint_sha256": _sha256_file(path),
        "source_tensor_key": source_key,
        "source_shape": list(tensor.shape),
        "source_dtype": str(tensor.dtype),
        "expected_oia_count": expected_count,
    }
    return tensor, report


def build_transformer_text_embeddings(
    names: Iterable[str],
    model_name_or_path: str,
    *,
    local_files_only: bool = True,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Encode predicate names with a frozen local Transformers text encoder."""
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    from transformers import AutoModel, AutoTokenizer

    texts = [name.replace("_", " ") for name in names]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    with torch.no_grad():
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        output = model(**encoded)
        hidden = output.last_hidden_state.float()
        mask = encoded["attention_mask"].float().unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        pooled = F.normalize(pooled, dim=-1)
    report = {
        "text_embedding_source": "transformers_frozen",
        "text_encoder_model": model_name_or_path,
        "text_embedding_shape": list(pooled.shape),
        "local_files_only": local_files_only,
    }
    return pooled, report


class TextPredicateTransfer(nn.Module):
    def __init__(
        self,
        predicate_names: Iterable[str],
        dim: int = 384,
        *,
        source_checkpoint: str | None = None,
        oia_predicate_names: Iterable[str] | None = None,
        text_encoder_model: str | None = None,
        require_source_checkpoint: bool = False,
        require_transformer_text: bool = False,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        names = list(predicate_names)
        self.predicate_names = names
        self.oia_predicate_names = list(oia_predicate_names or names[:32])
        self.oia_count = len(self.oia_predicate_names)
        if self.oia_count != 32:
            raise ValueError(f"Expected exact 32 OIA predicate names, got {self.oia_count}")
        if names[:32] != self.oia_predicate_names:
            raise ValueError("First 32 InteractFlow predicates must exactly match OIA predicate order")
        self.source_report: dict[str, object] = {
            "loaded_predicate_names": self.oia_predicate_names,
            "predicate_count_total": len(names),
            "oia_name_order_verified": True,
            "source_loaded": False,
        }
        try:
            if text_encoder_model:
                proto, text_report = build_transformer_text_embeddings(
                    names,
                    text_encoder_model,
                    local_files_only=local_files_only,
                )
            else:
                raise RuntimeError("No text_encoder_model configured")
        except Exception as exc:
            if require_transformer_text:
                raise
            proto = build_bow_text_embeddings(names, dim=dim)
            text_report = {
                "text_embedding_source": "ontology_bow_fallback",
                "text_encoder_model": text_encoder_model,
                "fallback_reason": repr(exc),
                "text_embedding_shape": list(proto.shape),
            }
        self.register_buffer("text_prototypes", proto.float())
        self.text_proj = nn.Linear(int(self.text_prototypes.shape[-1]), dim)
        self.oia_query_proj = nn.Linear(dim, dim, bias=False)
        if source_checkpoint:
            source, source_report = load_oia_predicate_queries(source_checkpoint, expected_count=self.oia_count)
            self.register_buffer("source_oia_queries", source.float())
            self.source_report.update(source_report)
            self.source_report["source_loaded"] = True
        elif require_source_checkpoint:
            raise ValueError("Formal PSI predicate transfer requires paths.oia_acpr_checkpoint")
        else:
            self.register_buffer("source_oia_queries", torch.zeros(self.oia_count, dim))
        self.oia_residual = nn.Parameter(torch.zeros(self.oia_count, dim))
        self.psi_residual = nn.Parameter(torch.zeros(max(0, len(names) - self.oia_count), dim))
        self.transfer_gate_raw = nn.Parameter(torch.zeros(len(names)))
        self.source_report.update(text_report)

    def forward(self, predicate_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        text = self.text_proj(self.text_prototypes.to(predicate_tokens.device, predicate_tokens.dtype))
        text = F.normalize(text, dim=-1)
        oia_source = self.oia_query_proj(self.source_oia_queries.to(predicate_tokens.device, predicate_tokens.dtype))
        oia_gate = torch.sigmoid(self.transfer_gate_raw[: self.oia_count]).to(predicate_tokens.device, predicate_tokens.dtype)
        oia_proto = text[: self.oia_count] + oia_gate[:, None] * oia_source + self.oia_residual.to(predicate_tokens.device, predicate_tokens.dtype)
        if len(self.predicate_names) > self.oia_count:
            psi_proto = text[self.oia_count :] + self.psi_residual.to(predicate_tokens.device, predicate_tokens.dtype)
            prototypes = torch.cat([oia_proto, psi_proto], dim=0)
        else:
            prototypes = oia_proto
        prototypes = F.normalize(prototypes, dim=-1)
        sim = torch.einsum("bpd,qd->bpq", predicate_tokens, prototypes)
        transferred = torch.einsum("bpq,qd->bpd", torch.softmax(sim, dim=-1), prototypes)
        gate = torch.sigmoid(self.transfer_gate_raw).to(predicate_tokens.device, predicate_tokens.dtype)
        return {
            "predicate_text_similarity": sim,
            "transferred_predicate_tokens": transferred,
            "transfer_gate": gate,
            "text_embedding_source": str(self.source_report.get("text_embedding_source", "unknown")),
            "source_loaded": torch.tensor(bool(self.source_report.get("source_loaded")), device=predicate_tokens.device),
        }

    def report(self) -> dict[str, object]:
        report = dict(self.source_report)
        report.update(
            {
                "mapped_shape": [len(self.predicate_names), int(self.oia_query_proj.out_features)],
                "transfer_gate_shape": [len(self.predicate_names)],
                "oia_transfer_formula": "W_o q_oia + W_n E_text(name) + residual",
            }
        )
        return report
