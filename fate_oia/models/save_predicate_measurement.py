from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .meter_schema import METERFactorSchema, default_meter_factor_schema
from .meter_signed_factors import (
    DEFAULT_STATE_CARDINALITIES,
    TypedEvidenceStateHead,
)


ACTION_PREDICATE_BRIDGE_SCALE = 0.05
REQUIRED_PREDICATE_OUTPUT_KEYS = (
    "predicate_map",
    "predicate_null_mass",
    "predicate_token",
    "predicate_state_prob",
    "predicate_state_entropy",
    "predicate_reliability",
    "predicate_groundable_mask",
    "predicate_named_mask",
    "predicate_mirror_pairs",
)


def fixed_action_predicate_bridge(value: Tensor) -> Tensor:
    """Expose a measurement tensor to action with an exact five-percent path."""
    return value.detach() + ACTION_PREDICATE_BRIDGE_SCALE * (
        value - value.detach()
    )


def selective_predicate_bridge(value: Tensor, *, scale: float = ACTION_PREDICATE_BRIDGE_SCALE) -> Tensor:
    """Backward-compatible name for the fixed action-to-predicate bridge."""
    if float(scale) != ACTION_PREDICATE_BRIDGE_SCALE:
        raise ValueError("SAVE action predicate bridge scale is fixed at 0.05")
    return fixed_action_predicate_bridge(value)


def _schema_masks(schema: METERFactorSchema) -> tuple[Tensor, Tensor]:
    groundable: list[float] = []
    named: list[float] = []
    for row in schema.rows:
        tier = str(row["groundability"]).lower()
        if tier == "full":
            groundable.append(1.0)
            named.append(1.0)
        elif tier == "partial":
            groundable.append(1.0)
            named.append(0.5)
        elif tier in {"latent", "none", "unavailable"}:
            groundable.append(0.0)
            named.append(0.0)
        else:
            raise ValueError(f"Unsupported SAVE predicate groundability: {tier}")
    return (
        torch.tensor(groundable, dtype=torch.float32),
        torch.tensor(named, dtype=torch.float32),
    )


class SAVEPredicateMeasurement(nn.Module):
    """SAVE's typed soft predicate measurement wrapper.

    The wrapped head owns the visual measurement parameters and detaches both
    foundation inputs.  Raw predicate values are used by train-only grounding;
    action consumers use the explicit ``*_action`` views below.
    """

    def __init__(
        self,
        dim: int = 384,
        factor_dim: int = 21,
        num_layers: int = 3,
        state_cardinalities: tuple[int, ...] = DEFAULT_STATE_CARDINALITIES,
        schema_path: str | Path | None = None,
        typed_head: TypedEvidenceStateHead | None = None,
    ) -> None:
        super().__init__()
        if typed_head is None:
            typed_head = TypedEvidenceStateHead(
                dim=dim,
                factor_dim=factor_dim,
                num_layers=num_layers,
                state_cardinalities=state_cardinalities,
                schema_path=str(schema_path) if schema_path is not None else None,
                action_measurement_grad_scale=0.0,
            )
        # SAVE owns the sole action-route bridge.  Disable the wrapped head's
        # legacy bridge so the effective parameter gradient is 0.05, not 0.0025.
        typed_head.action_measurement_grad_scale = 0.0
        self.typed_evidence_state_head = typed_head

        if schema_path is not None:
            schema = METERFactorSchema(schema_path)
        elif typed_head.factor_dim == 21:
            schema = default_meter_factor_schema()
        else:
            schema = None
        if schema is not None:
            if len(schema.rows) != typed_head.factor_dim:
                raise ValueError("SAVE schema factor count does not match typed head")
            groundable_mask, named_mask = _schema_masks(schema)
            mirror_pairs = schema.mirror_pairs
            self.schema_sha256 = schema.sha256
        else:
            groundable_mask = typed_head.groundable_mask.detach().clone()
            named_mask = groundable_mask.clone()
            mirror_pairs = tuple(typed_head.mirror_pairs)
            self.schema_sha256 = ""
        self.register_buffer(
            "predicate_groundable_mask", groundable_mask, persistent=True
        )
        self.register_buffer("predicate_named_mask", named_mask, persistent=True)
        self.action_bridge_scale = ACTION_PREDICATE_BRIDGE_SCALE
        self.predicate_mirror_pairs = tuple(mirror_pairs)

    @property
    def typed_head(self) -> TypedEvidenceStateHead:
        """Return the single registered typed measurement owner."""
        return self.typed_evidence_state_head

    @property
    def groundable_mask(self) -> Tensor:
        return self.predicate_groundable_mask

    @property
    def named_mask(self) -> Tensor:
        return self.predicate_named_mask

    def forward(
        self,
        factor_base_nodes: Tensor,
        patch_tokens_by_layer: Tensor,
        progress: float,
    ) -> dict[str, Tensor | tuple[tuple[int, int], ...] | str]:
        typed = self.typed_evidence_state_head(
            factor_base_nodes,
            patch_tokens_by_layer,
            progress=progress,
        )
        predicate_map_raw = typed["factor_anchor_map"]
        predicate_token_raw = typed["factor_typed_token"]
        predicate_state_prob_raw = typed["factor_state_prob"]
        predicate_reliability_raw = typed["factor_reliability"]

        predicate_map_action = fixed_action_predicate_bridge(predicate_map_raw)
        predicate_token_action = fixed_action_predicate_bridge(predicate_token_raw)
        predicate_state_prob_action = fixed_action_predicate_bridge(
            predicate_state_prob_raw
        )
        predicate_reliability_action = fixed_action_predicate_bridge(
            predicate_reliability_raw
        )
        output: dict[str, Tensor | tuple[tuple[int, int], ...] | str] = {
            # The unsuffixed values are the action-facing contract.  Raw
            # values remain available for grounding and gradient audits.
            "predicate_map": predicate_map_action,
            "predicate_map_raw": predicate_map_raw,
            "predicate_null_mass": typed["factor_null_mass"],
            "predicate_null_logit": typed["factor_null_logit"],
            "predicate_anchor_token": typed["factor_anchor_token"],
            "predicate_global_token": typed["factor_global_token"],
            "predicate_token": predicate_token_action,
            "predicate_token_raw": predicate_token_raw,
            "predicate_typed_token": predicate_token_raw,
            "predicate_state_logits": typed["factor_state_logits"],
            "predicate_state_prob": predicate_state_prob_action,
            "predicate_state_prob_raw": predicate_state_prob_raw,
            "predicate_state_entropy": typed["factor_state_entropy"],
            "predicate_state_valid_mask": typed["factor_state_valid_mask"],
            "predicate_state_cardinalities": self.typed_evidence_state_head.state_cardinalities,
            "predicate_visual_confidence": typed["factor_visual_confidence"],
            "predicate_reliability": predicate_reliability_action,
            "predicate_reliability_raw": predicate_reliability_raw,
            "predicate_groundable_mask": self.predicate_groundable_mask,
            "predicate_named_mask": self.predicate_named_mask,
            "predicate_mirror_pairs": self.predicate_mirror_pairs,
            "predicate_schema_sha256": self.schema_sha256,
            "predicate_map_action": predicate_map_action,
            "predicate_token_action": predicate_token_action,
            "predicate_state_prob_action": predicate_state_prob_action,
            "predicate_reliability_action": predicate_reliability_action,
            # Prefix aliases match action-reader naming used by SAVE callers.
            "predicate_action_map": predicate_map_action,
            "predicate_action_token": predicate_token_action,
            "predicate_action_state_prob": predicate_state_prob_action,
            "predicate_action_reliability": predicate_reliability_action,
            "predicate_ontology_query": typed["factor_ontology_query"],
            "predicate_ontology_target": typed["factor_ontology_target"],
            "predicate_state_ontology_query": typed["state_ontology_query"],
            "predicate_state_ontology_target": typed["state_ontology_target"],
            "predicate_factor_group_ids": typed["factor_group_ids"],
        }
        return output


__all__ = [
    "ACTION_PREDICATE_BRIDGE_SCALE",
    "REQUIRED_PREDICATE_OUTPUT_KEYS",
    "SAVEPredicateMeasurement",
    "fixed_action_predicate_bridge",
    "selective_predicate_bridge",
]
