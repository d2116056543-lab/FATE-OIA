"""Compositional, action-safe semantic reason queries for RAEL-OIA P4.

Semantic queries use only the static 21-row ontology and image-derived visual
fields.  Future P5 ledger tokens cross the explicit ``EvidenceReadBundle``
boundary; they are not substituted by labels or geometry.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch import Tensor, nn

from fate_oia.models.rael_multilayer_field import RAELMultiLayerField
from fate_oia.utils.rael_schema import ROLE_NAMES, ReasonSemanticRow, load_reason_semantic_schema


@runtime_checkable
class MultiLayerFieldReader(Protocol):
    """P3 reading surface shared by semantic/action/ledger consumers."""

    def read(
        self,
        prepared: Mapping[str, Any],
        queries: Tensor,
        group_name: str | None = None,
    ) -> Mapping[str, Tensor | str | None]: ...


@dataclass(frozen=True)
class EvidenceReadBundle:
    """Future-ledger evidence supplied as image-derived token content only."""

    tokens: Tensor
    valid_mask: Tensor | None = None

    def __post_init__(self) -> None:
        self._validate_finite_inputs()

    def _validate_finite_inputs(self) -> None:
        if not torch.is_tensor(self.tokens):
            raise ValueError("evidence tokens must be a Tensor")
        if not bool(torch.isfinite(self.tokens).all()):
            raise ValueError("evidence tokens must be finite")
        if self.valid_mask is not None:
            if not torch.is_tensor(self.valid_mask):
                raise ValueError("evidence valid_mask must be a Tensor")
            if not bool(torch.isfinite(self.valid_mask).all()):
                raise ValueError("evidence valid_mask must be finite")

    def validate(self, batch_size: int, dim: int, device: torch.device) -> tuple[Tensor, Tensor]:
        # Tensors can be modified after construction, so validate the boundary again.
        self._validate_finite_inputs()
        if not torch.is_tensor(self.tokens) or self.tokens.ndim != 3:
            raise ValueError("evidence tokens must be [B,K,D]")
        if self.tokens.shape[0] != batch_size or self.tokens.shape[2] != dim:
            raise ValueError("evidence tokens must match semantic query batch and dimension")
        if self.tokens.shape[1] <= 0:
            raise ValueError("evidence tokens must include at least one token")
        if self.tokens.device != device:
            raise ValueError("evidence tokens and semantic queries must use the same device")
        if self.valid_mask is None:
            mask = torch.ones(self.tokens.shape[:2], dtype=torch.bool, device=device)
        else:
            mask = self.valid_mask
            if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != self.tokens.shape[:2]:
                raise ValueError("evidence valid_mask must be bool [B,K]")
            if mask.device != device:
                raise ValueError("evidence valid_mask and semantic queries must use the same device")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every sample must provide at least one valid evidence token")
        return self.tokens, mask


class RAELSemanticReason(nn.Module):
    """Read compositional reason semantics from P3 fields and ledger evidence."""

    FORMAL_REASON_COUNT = 21
    FORMAL_DIM = 384
    parameter_owner = "semantic_reason"

    def __init__(self, schema_path: str | Path, dim: int = FORMAL_DIM) -> None:
        super().__init__()
        if dim != self.FORMAL_DIM:
            raise ValueError(f"RAEL semantic reason requires dim={self.FORMAL_DIM}")
        self.dim = int(dim)
        self.rows: tuple[ReasonSemanticRow, ...] = load_reason_semantic_schema(schema_path)
        if len(self.rows) != self.FORMAL_REASON_COUNT:
            raise ValueError("RAEL semantic reason requires exactly 21 schema rows")

        entity_vocab = self._vocabulary(row.entity for row in self.rows)
        state_vocab = self._vocabulary(row.state for row in self.rows)
        sector_vocab = self._vocabulary(row.sector for row in self.rows)
        role_vocab = self._vocabulary(row.role for row in self.rows)
        self.entity_embedding = nn.Embedding(len(entity_vocab), self.dim)
        self.state_embedding = nn.Embedding(len(state_vocab), self.dim)
        self.sector_embedding = nn.Embedding(len(sector_vocab), self.dim)
        self.role_embedding = nn.Embedding(len(role_vocab), self.dim)
        self.query_norm = nn.LayerNorm(self.dim)
        self.reason_residual = nn.Parameter(torch.zeros(self.FORMAL_REASON_COUNT, self.dim))

        self.evidence_query_projection = nn.Linear(self.dim, self.dim, bias=False)
        self.evidence_key_projection = nn.Linear(self.dim, self.dim, bias=False)
        self.evidence_value_projection = nn.Linear(self.dim, self.dim, bias=False)
        self.evidence_gate = nn.Parameter(torch.zeros(self.FORMAL_REASON_COUNT))
        self.output_norm = nn.LayerNorm(self.dim)

        self.register_buffer("entity_ids", self._ids(self.rows, "entity", entity_vocab), persistent=True)
        self.register_buffer("state_ids", self._ids(self.rows, "state", state_vocab), persistent=True)
        self.register_buffer("sector_ids", self._ids(self.rows, "sector", sector_vocab), persistent=True)
        self.register_buffer("role_ids", self._ids(self.rows, "role", role_vocab), persistent=True)
        self.reset_parameters()

    @staticmethod
    def _vocabulary(values: Any) -> dict[str, int]:
        return {value: index for index, value in enumerate(sorted(set(values)))}

    @staticmethod
    def _ids(rows: tuple[ReasonSemanticRow, ...], field: str, vocabulary: Mapping[str, int]) -> Tensor:
        return torch.tensor([vocabulary[getattr(row, field)] for row in rows], dtype=torch.long)

    def reset_parameters(self) -> None:
        for embedding in (
            self.entity_embedding,
            self.state_embedding,
            self.sector_embedding,
            self.role_embedding,
        ):
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        for projection in (
            self.evidence_query_projection,
            self.evidence_key_projection,
            self.evidence_value_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)
        nn.init.zeros_(self.reason_residual)
        nn.init.zeros_(self.evidence_gate)
        nn.init.ones_(self.query_norm.weight)
        nn.init.zeros_(self.query_norm.bias)
        nn.init.ones_(self.output_norm.weight)
        nn.init.zeros_(self.output_norm.bias)

    def owned_parameter_names(self) -> tuple[str, ...]:
        """Expose a stable owner boundary for the P13 gradient admission stage."""

        return tuple(name for name, _ in self.named_parameters())

    def _parameter_dtype(self) -> torch.dtype:
        return next(self.entity_embedding.parameters()).dtype

    def compositional_queries(self, batch_size: int) -> Tensor:
        """Return q=LN(entity+state+sector+role)+0.1*tanh(residual)."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        base = self.query_norm(
            self.entity_embedding(self.entity_ids)
            + self.state_embedding(self.state_ids)
            + self.sector_embedding(self.sector_ids)
            + self.role_embedding(self.role_ids)
        )
        query = base + 0.1 * torch.tanh(self.reason_residual)
        return query.unsqueeze(0).expand(batch_size, -1, -1)

    def read(
        self,
        field_reader: RAELMultiLayerField | MultiLayerFieldReader,
        prepared_field: Mapping[str, Any],
        evidence: EvidenceReadBundle,
    ) -> dict[str, Tensor]:
        """Compute S_r=Read(q_r,F,E) without supervision-time inputs.

        ``field_reader`` is the shared P3 reader.  ``evidence`` is a typed
        future-ledger token bundle; both are required so this cannot degrade to
        an untracked field-only path.
        """

        if not isinstance(evidence, EvidenceReadBundle):
            raise TypeError("evidence must be an EvidenceReadBundle")
        if not isinstance(field_reader, MultiLayerFieldReader):
            raise TypeError("field_reader must provide the P3 read protocol")
        batch_size: int | None = None
        if torch.is_tensor(evidence.tokens):
            batch_size = int(evidence.tokens.shape[0])
        if batch_size is None or batch_size <= 0:
            raise ValueError("evidence must provide a positive batch size")
        queries = self.compositional_queries(batch_size)
        field_read = field_reader.read(prepared_field, queries, group_name="semantic_reason")
        if not isinstance(field_read, Mapping):
            raise TypeError("P3 read must return a mapping")
        field_tokens = field_read.get("readout")
        layer_weights = field_read.get("layer_weights")
        if not torch.is_tensor(field_tokens) or field_tokens.shape != (batch_size, self.FORMAL_REASON_COUNT, self.dim):
            raise ValueError("P3 semantic readout must be [B,21,384]")
        if not torch.is_tensor(layer_weights) or layer_weights.shape != (batch_size, self.FORMAL_REASON_COUNT, 4):
            raise ValueError("P3 semantic layer weights must be [B,21,4]")
        # P3 may legitimately use its internal CUDA BF16 path.  P4 owns FP32
        # projections by default, so normalize the local read boundary rather
        # than relying on the caller to nest autocast contexts correctly.
        field_tokens = field_tokens.to(dtype=self._parameter_dtype())
        tokens, valid_mask = evidence.validate(batch_size, self.dim, field_tokens.device)
        tokens = tokens.to(dtype=field_tokens.dtype)

        evidence_query = self.evidence_query_projection(queries.to(dtype=field_tokens.dtype))
        evidence_keys = self.evidence_key_projection(tokens)
        evidence_values = self.evidence_value_projection(tokens)
        evidence_scores = torch.einsum("brd,bkd->brk", evidence_query, evidence_keys) * (self.dim**-0.5)
        evidence_scores = evidence_scores.masked_fill(~valid_mask.unsqueeze(1), torch.finfo(evidence_scores.dtype).min)
        evidence_weights = torch.softmax(evidence_scores, dim=-1)
        evidence_readout = torch.einsum("brk,bkd->brd", evidence_weights, evidence_values)
        gate = torch.sigmoid(self.evidence_gate).view(1, self.FORMAL_REASON_COUNT, 1)
        semantic_tokens = self.output_norm(field_tokens + gate * evidence_readout)
        return {
            "semantic_reason_tokens": semantic_tokens,
            "semantic_queries": queries,
            "layer_weights": layer_weights,
            "evidence_weights": evidence_weights,
            "evidence_gate": gate.expand(batch_size, -1, -1),
        }

    def forward(
        self,
        field_reader: RAELMultiLayerField | MultiLayerFieldReader,
        prepared_field: Mapping[str, Any],
        evidence: EvidenceReadBundle,
    ) -> dict[str, Tensor]:
        return self.read(field_reader, prepared_field, evidence)


__all__ = ["EvidenceReadBundle", "MultiLayerFieldReader", "RAELSemanticReason", "ROLE_NAMES"]
