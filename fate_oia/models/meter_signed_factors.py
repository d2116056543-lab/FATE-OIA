from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .acpr_sparse_ops import entmax15_bisect
from .meter_schema import METERFactorSchema, default_meter_factor_schema


DEFAULT_STATE_CARDINALITIES = (3,) * 21
DEFAULT_ACTION_OWNERSHIP = (
    1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
)
DEFAULT_GROUNDABLE = (
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0,
)


def selective_credit_bridge(
    anchor: Tensor,
    state: Tensor,
    global_token: Tensor,
    *,
    scale: float = 0.05,
) -> Tensor:
    """Expose semantic evidence to action without letting action move anchors."""
    if not 0.0 <= scale <= 0.20:
        raise ValueError("HECA measurement bridge scale must be in [0, 0.20]")
    state_bridge = state.detach() + scale * (state - state.detach())
    global_bridge = global_token.detach() + scale * (
        global_token - global_token.detach()
    )
    return torch.cat((anchor.detach(), state_bridge, global_bridge), dim=-1)


class TypedEvidenceStateHead(nn.Module):
    """Factor-specific anchor, state, observability, and typed evidence."""

    def __init__(
        self,
        dim: int = 384,
        factor_dim: int = 21,
        num_layers: int = 3,
        state_cardinalities: tuple[int, ...] = DEFAULT_STATE_CARDINALITIES,
        schema_path: str | None = None,
        anchor_exploration_mass: float = 0.05,
        action_measurement_grad_scale: float = 0.05,
        factor_text_prototype_path: str | None = None,
        state_text_prototype_path: str | None = None,
    ) -> None:
        super().__init__()
        schema = METERFactorSchema(schema_path) if schema_path else default_meter_factor_schema()
        if len(schema.rows) != factor_dim:
            raise ValueError("METER schema factor count does not match factor_dim")
        if schema_path is None and state_cardinalities != DEFAULT_STATE_CARDINALITIES:
            # Preserve the legacy unit-test override; production construction
            # always uses the YAML schema as the source of truth.
            state_cardinalities = tuple(state_cardinalities)
        else:
            state_cardinalities = schema.state_cardinalities
        if len(state_cardinalities) != factor_dim:
            raise ValueError("One state cardinality is required per factor")
        self.dim = int(dim)
        self.factor_dim = int(factor_dim)
        self.num_layers = int(num_layers)
        self.anchor_exploration_mass = float(anchor_exploration_mass)
        self.action_measurement_grad_scale = float(action_measurement_grad_scale)
        if not 0.0 <= self.anchor_exploration_mass < 1.0:
            raise ValueError("anchor_exploration_mass must be in [0, 1)")
        if not 0.0 <= self.action_measurement_grad_scale <= 0.20:
            raise ValueError("action_measurement_grad_scale must be in [0, 0.20]")
        self.max_states = max(state_cardinalities)
        self.register_buffer(
            "state_cardinalities",
            torch.tensor(state_cardinalities, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "action_ownership",
            torch.tensor(schema.action_ownership, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "groundable_mask",
            torch.tensor(schema.groundable_mask, dtype=torch.float32),
            persistent=True,
        )
        self.anchor_query = nn.Linear(dim, dim)
        self.factor_semantic_query = nn.Parameter(torch.randn(factor_dim, dim) * 0.02)
        self.factor_spatial_query = nn.Parameter(torch.randn(factor_dim, dim) * 0.02)
        factor_text = self._load_prototype(
            factor_text_prototype_path, (factor_dim,), fallback_dim=dim
        )
        group_names = sorted({str(row["factor_group"]) for row in schema.rows})
        self.register_buffer(
            "factor_group_ids",
            torch.tensor(
                [group_names.index(str(row["factor_group"])) for row in schema.rows],
                dtype=torch.long,
            ),
            persistent=True,
        )
        state_text = self._load_prototype(
            state_text_prototype_path,
            (factor_dim, self.max_states),
            fallback_dim=factor_text.shape[-1],
        )
        if factor_text.shape[-1] != state_text.shape[-1]:
            raise ValueError("Factor/state ontology prototype dimensions differ")
        self.register_buffer("factor_text_prototype", factor_text, persistent=True)
        self.register_buffer("state_text_prototype", state_text, persistent=True)
        self.factor_text_proj = nn.Linear(factor_text.shape[-1], dim, bias=False)
        self.state_text_proj = nn.Linear(state_text.shape[-1], dim, bias=False)
        self.anchor_key = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.anchor_value = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.layer_router = nn.Parameter(torch.zeros(factor_dim, num_layers))
        # The null score is calibrated against each patch partition, not held
        # at an absolute logit where sparse attention can permanently remove it.
        self.null_bias = nn.Parameter(torch.zeros(factor_dim))
        self.global_proj = nn.Linear(dim, dim)
        self.anchor_proj = nn.Linear(dim, dim)
        self.action_anchor_proj = nn.Linear(dim, dim)
        self.state_embeddings = nn.Parameter(
            torch.randn(factor_dim, self.max_states, dim) * 0.02
        )
        self.action_state_embeddings = nn.Parameter(
            torch.randn(factor_dim, self.max_states, dim) * 0.02
        )
        self.state_weight = nn.Parameter(
            torch.randn(factor_dim, self.max_states, dim * 3) * 0.02
        )
        self.state_bias = nn.Parameter(torch.zeros(factor_dim, self.max_states))
        self.typed_norm = nn.LayerNorm(dim)
        self.action_route_norm = nn.LayerNorm(dim)
        self.action_value_norm = nn.LayerNorm(dim)
        self.action_bridge_proj = nn.Linear(dim * 3, dim)
        self.schema_sha256 = schema.sha256
        self.mirror_pairs = schema.mirror_pairs

    @staticmethod
    def _load_prototype(
        path: str | None,
        leading_shape: tuple[int, ...],
        *,
        fallback_dim: int,
    ) -> Tensor:
        if path is None:
            return torch.zeros((*leading_shape, fallback_dim), dtype=torch.float32)
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Missing offline HECA ontology prototype: {source}")
        value = torch.load(source, map_location="cpu", weights_only=True)
        if isinstance(value, dict):
            value = value.get("prototype")
        value = torch.as_tensor(value, dtype=torch.float32)
        if tuple(value.shape[:-1]) != leading_shape:
            raise ValueError(
                f"HECA ontology prototype leading shape {tuple(value.shape[:-1])} != {leading_shape}"
            )
        return value

    @staticmethod
    def _ramp(progress: float) -> float:
        return float(min(max(progress / 0.20, 0.0), 1.0))

    def _distribution(self, logits: Tensor, progress: float) -> Tensor:
        dense = torch.softmax(logits, dim=-1)
        sparse = entmax15_bisect(logits, dim=-1)
        ramp = self._ramp(progress)
        scheduled = dense * (1.0 - ramp) + sparse * ramp
        exploration = self.anchor_exploration_mass * max(1.0 - progress / 0.20, 0.0)
        return scheduled * (1.0 - exploration) + dense * exploration

    def _partition_calibrated_null(self, patch_logits: Tensor) -> tuple[Tensor, Tensor]:
        """Return null probability relative to the patch log-mean-exp partition."""
        log_mean_exp = torch.logsumexp(patch_logits, dim=-1) - math.log(
            patch_logits.shape[-1]
        )
        mean = patch_logits.mean(dim=-1)
        null_logit = self.null_bias.view(1, -1) + mean - log_mean_exp
        return torch.sigmoid(null_logit), null_logit

    def compose_typed_token(
        self,
        global_token: Tensor,
        anchor_token: Tensor,
        state_prob: Tensor,
    ) -> Tensor:
        state_basis = self.state_embeddings + 0.20 * self.state_text_proj(
            self.state_text_prototype
        )
        state_token = torch.einsum("bfs,fsd->bfd", state_prob, state_basis)
        return self.typed_norm(
            global_token + self.anchor_proj(anchor_token) + state_token
        )

    def compose_action_token(
        self,
        anchor_token: Tensor,
        state_prob: Tensor,
    ) -> Tensor:
        """Compose action routing evidence without the global semantic shortcut."""
        # Action transport may learn how to read shared evidence, but it must not
        # rewrite the anchor/state representation used by the reason branch.
        anchor_token = anchor_token.detach()
        state_prob = state_prob.detach()
        state_basis = self.action_state_embeddings + 0.20 * self.state_text_proj(
            self.state_text_prototype
        )
        state_token = torch.einsum("bfs,fsd->bfd", state_prob, state_basis)
        return self.action_route_norm(
            self.action_anchor_proj(anchor_token) + state_token
        )

    def compose_action_bridge_token(
        self,
        anchor_token: Tensor,
        state_prob: Tensor,
        global_token: Tensor,
    ) -> tuple[Tensor, Tensor]:
        state_basis = self.action_state_embeddings + 0.20 * self.state_text_proj(
            self.state_text_prototype
        )
        state_token = torch.einsum("bfs,fsd->bfd", state_prob, state_basis)
        bridge = selective_credit_bridge(
            anchor_token,
            state_token,
            global_token,
            scale=self.action_measurement_grad_scale,
        )
        state_prob_credit = state_prob.detach() + self.action_measurement_grad_scale * (
            state_prob - state_prob.detach()
        )
        return self.action_bridge_proj(bridge), state_prob_credit

    def compose_action_value_token(self, anchor_token: Tensor) -> Tensor:
        """Keep the transported action value causally tied to local patches."""
        return self.action_value_norm(
            self.action_anchor_proj(anchor_token.detach())
        )

    def _state_posterior(self, state_input: Tensor) -> tuple[Tensor, Tensor]:
        logits = torch.einsum("bfd,fsd->bfs", state_input, self.state_weight) + self.state_bias
        state_index = torch.arange(self.max_states, device=logits.device)
        valid = state_index.view(1, -1) < self.state_cardinalities.view(-1, 1)
        logits = logits.masked_fill(~valid.unsqueeze(0), float("-inf"))
        return logits, torch.softmax(logits, dim=-1)

    def forward(
        self,
        factor_base_nodes: Tensor,
        patch_tokens_by_layer: Tensor,
        *,
        progress: float = 1.0,
    ) -> dict[str, Tensor]:
        if factor_base_nodes.shape[1:] != (self.factor_dim, self.dim):
            raise ValueError("Expected factor_base_nodes [B,21,D]")
        if patch_tokens_by_layer.ndim != 4:
            raise ValueError("Expected patch_tokens_by_layer [B,S,N,D]")
        # Typed factors own their parameters. Grounding supervision must not
        # rewrite the CalAlign-compatible visual foundation.
        factor_base_nodes = factor_base_nodes.detach()
        patch_tokens_by_layer = patch_tokens_by_layer.detach()
        query_source = (
            factor_base_nodes
            + self.factor_semantic_query.unsqueeze(0)
            + self.factor_spatial_query.unsqueeze(0)
            + 0.20 * self.factor_text_proj(self.factor_text_prototype).unsqueeze(0)
        )
        query = self.anchor_query(query_source)
        layer_weight = torch.softmax(self.layer_router, dim=-1)
        maps: list[Tensor] = []
        nulls: list[Tensor] = []
        tokens: list[Tensor] = []
        for layer in range(self.num_layers):
            patches = patch_tokens_by_layer[:, layer]
            key = self.anchor_key[layer](patches)
            value = self.anchor_value[layer](patches)
            patch_logits = torch.einsum("bfd,bnd->bfn", query, key) / math.sqrt(
                self.dim
            )
            null_mass, _ = self._partition_calibrated_null(patch_logits)
            conditional_patch = self._distribution(patch_logits, progress)
            anchored_patch = (1.0 - null_mass).unsqueeze(-1) * conditional_patch
            maps.append(anchored_patch)
            nulls.append(null_mass)
            tokens.append(torch.einsum("bfn,bnd->bfd", anchored_patch, value))
        anchor_map = torch.einsum("fs,bfsn->bfn", layer_weight, torch.stack(maps, 2))
        null_mass = torch.einsum("fs,bfs->bf", layer_weight, torch.stack(nulls, 2))
        null_logit = torch.logit(null_mass.clamp(1e-6, 1.0 - 1e-6))
        anchor_token = torch.einsum(
            "fs,bfsd->bfd", layer_weight, torch.stack(tokens, 2)
        )
        global_token = self.global_proj(factor_base_nodes)
        pooled_global = patch_tokens_by_layer.mean(dim=(1, 2))
        state_input = torch.cat(
            [
                global_token,
                anchor_token,
                pooled_global.unsqueeze(1).expand(-1, self.factor_dim, -1),
            ],
            dim=-1,
        )
        state_logits, state_prob = self._state_posterior(state_input)
        # Action reads the same posterior values, but its permitted bridge may
        # train state parameters without changing anchor/global measurement.
        action_state_logits, action_state_prob = self._state_posterior(
            state_input.detach()
        )
        state_index = torch.arange(self.max_states, device=state_logits.device)
        state_valid_mask = state_index.view(1, -1) < self.state_cardinalities.view(
            -1, 1
        )
        entropy = -(state_prob.clamp_min(1e-8).log() * state_prob).sum(-1)
        entropy_norm = entropy / self.state_cardinalities.to(
            state_prob.dtype
        ).log().clamp_min(1e-6).view(1, -1)
        # BDD100K source availability is a training-only provenance condition.
        # It is not identifiable from the matched image when a factor's source
        # label is one-class, so it must not be a learned gate in test forward.
        # Visual confidence is instead fully determined by image-derived null
        # evidence and state uncertainty.
        visual_confidence = (
            (1.0 - null_mass) * (1.0 - entropy_norm)
        ).clamp(0.0, 1.0)
        typed_token = self.compose_typed_token(
            global_token, anchor_token, state_prob
        )
        action_token = self.compose_action_token(anchor_token, state_prob)
        action_value_token = self.compose_action_value_token(anchor_token)
        action_bridge_token, state_prob_credit = self.compose_action_bridge_token(
            anchor_token, action_state_prob, global_token
        )
        return {
            "factor_anchor_map": anchor_map,
            "factor_null_mass": null_mass,
            "factor_null_logit": null_logit,
            "factor_anchor_token": anchor_token,
            "factor_global_token": global_token,
            "factor_state_logits": state_logits,
            "factor_state_prob": state_prob,
            "factor_state_logits_action": action_state_logits,
            "factor_state_prob_action": action_state_prob,
            "factor_state_valid_mask": state_valid_mask,
            "factor_state_entropy": entropy,
            "factor_visual_confidence": visual_confidence,
            # Compatibility aliases for legacy readers. These are derived from
            # visual evidence only and have no learned source-availability head.
            "factor_observability_logit": torch.logit(
                visual_confidence.clamp(1e-6, 1.0 - 1e-6)
            ),
            "factor_observability": visual_confidence,
            "factor_reliability": visual_confidence,
            "factor_typed_token": typed_token,
            "factor_action_token": action_token,
            "factor_action_value_token": action_value_token,
            "factor_action_bridge_token": action_bridge_token,
            "factor_state_prob_credit": state_prob_credit,
            "factor_measurement_token": typed_token,
            "factor_ontology_query": self.factor_semantic_query,
            "factor_ontology_target": self.factor_text_proj(self.factor_text_prototype),
            "state_ontology_query": self.state_embeddings,
            "state_ontology_target": self.state_text_proj(self.state_text_prototype),
            "factor_layer_weights": layer_weight,
            "factor_action_ownership": self.action_ownership,
            "factor_groundable_mask": self.groundable_mask,
            "factor_group_ids": self.factor_group_ids,
        }
