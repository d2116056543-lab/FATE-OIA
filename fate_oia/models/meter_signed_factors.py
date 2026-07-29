from __future__ import annotations

import math

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


class TypedEvidenceStateHead(nn.Module):
    """Factor-specific anchor, state, observability, and typed evidence."""

    def __init__(
        self,
        dim: int = 384,
        factor_dim: int = 21,
        num_layers: int = 3,
        state_cardinalities: tuple[int, ...] = DEFAULT_STATE_CARDINALITIES,
        schema_path: str | None = None,
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
        self.anchor_key = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.anchor_value = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.layer_router = nn.Parameter(torch.zeros(factor_dim, num_layers))
        self.null_bias = nn.Parameter(torch.full((factor_dim,), math.log(0.1 / 0.9)))
        self.global_proj = nn.Linear(dim, dim)
        self.anchor_proj = nn.Linear(dim, dim)
        self.state_embeddings = nn.Parameter(
            torch.randn(factor_dim, self.max_states, dim) * 0.02
        )
        self.state_weight = nn.Parameter(
            torch.randn(factor_dim, self.max_states, dim * 3) * 0.02
        )
        self.state_bias = nn.Parameter(torch.zeros(factor_dim, self.max_states))
        self.obs_head = nn.Parameter(torch.randn(factor_dim, dim * 2 + 2) * 0.02)
        self.obs_bias = nn.Parameter(torch.zeros(factor_dim))
        self.typed_norm = nn.LayerNorm(dim)
        self.schema_sha256 = schema.sha256
        self.mirror_pairs = schema.mirror_pairs

    @staticmethod
    def _ramp(progress: float) -> float:
        return float(min(max(progress / 0.10, 0.0), 1.0))

    def _distribution(self, logits: Tensor, progress: float) -> Tensor:
        dense = torch.softmax(logits, dim=-1)
        sparse = entmax15_bisect(logits, dim=-1)
        ramp = self._ramp(progress)
        return dense * (1.0 - ramp) + sparse * ramp

    def compose_typed_token(
        self,
        global_token: Tensor,
        anchor_token: Tensor,
        state_prob: Tensor,
    ) -> Tensor:
        state_token = torch.einsum(
            "bfs,fsd->bfd", state_prob, self.state_embeddings
        )
        return self.typed_norm(
            global_token + self.anchor_proj(anchor_token) + state_token
        )

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
        query = self.anchor_query(factor_base_nodes)
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
            null_logits = self.null_bias.view(1, -1, 1).expand(
                patch_logits.shape[0], -1, 1
            )
            full = self._distribution(
                torch.cat([patch_logits, null_logits], dim=-1), progress
            )
            maps.append(full[..., :-1])
            nulls.append(full[..., -1])
            tokens.append(torch.einsum("bfn,bnd->bfd", full[..., :-1], value))
        anchor_map = torch.einsum("fs,bfsn->bfn", layer_weight, torch.stack(maps, 2))
        null_mass = torch.einsum("fs,bfs->bf", layer_weight, torch.stack(nulls, 2))
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
        state_logits = torch.einsum(
            "bfd,fsd->bfs", state_input, self.state_weight
        ) + self.state_bias
        state_index = torch.arange(self.max_states, device=state_logits.device)
        state_valid_mask = state_index.view(1, -1) < self.state_cardinalities.view(
            -1, 1
        )
        state_logits = state_logits.masked_fill(
            ~state_valid_mask.unsqueeze(0), float("-inf")
        )
        state_prob = torch.softmax(state_logits, dim=-1)
        entropy = -(state_prob.clamp_min(1e-8).log() * state_prob).sum(-1)
        entropy_norm = entropy / self.state_cardinalities.to(
            state_prob.dtype
        ).log().clamp_min(1e-6).view(1, -1)
        obs_input = torch.cat(
            [global_token, anchor_token, null_mass.unsqueeze(-1), entropy.unsqueeze(-1)],
            dim=-1,
        )
        obs_logit = torch.einsum("bfd,fd->bf", obs_input, self.obs_head) + self.obs_bias
        observability = torch.sigmoid(obs_logit)
        reliability = (
            observability * (1.0 - null_mass) * (1.0 - entropy_norm)
        ).clamp(0.0, 1.0)
        typed_token = self.compose_typed_token(
            global_token, anchor_token, state_prob
        )
        return {
            "factor_anchor_map": anchor_map,
            "factor_null_mass": null_mass,
            "factor_anchor_token": anchor_token,
            "factor_global_token": global_token,
            "factor_state_logits": state_logits,
            "factor_state_prob": state_prob,
            "factor_state_valid_mask": state_valid_mask,
            "factor_state_entropy": entropy,
            "factor_observability_logit": obs_logit,
            "factor_observability": observability,
            "factor_reliability": reliability,
            "factor_typed_token": typed_token,
            "factor_layer_weights": layer_weight,
            "factor_action_ownership": self.action_ownership,
            "factor_groundable_mask": self.groundable_mask,
        }


# Compatibility import name; the implementation is V2 typed evidence.
METERsignedFactors = TypedEvidenceStateHead
