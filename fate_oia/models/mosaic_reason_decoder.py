from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from .mosaic_sparse_label_decoder import MOSAICSparseLabelDecoder


class MOSAICReasonDecoder(nn.Module):
    def __init__(
        self,
        factor_names: Sequence[str],
        states: dict[str, Any],
        reason_observation: dict[int, Any],
        *,
        dim: int = 384,
        decoder_layers: int = 2,
        self_attention_heads: int = 4,
        highres_topk: int = 256,
        midres_topk: int = 128,
    ) -> None:
        super().__init__()
        self.factor_names = tuple(factor_names)
        self.factor_index = {name: index for index, name in enumerate(self.factor_names)}
        self.state_names = tuple(states)
        self.state_index = {name: index for index, name in enumerate(self.state_names)}
        self.dim = dim
        if set(reason_observation) != set(range(21)):
            raise ValueError("reason decoder requires observations for reason indices 0..20")
        reason_factor_map = torch.zeros(21, len(self.factor_names), dtype=torch.bool)
        state_factor_closure = {
            state_name: self._positive_state_factors(state_name, states, set()) for state_name in self.state_names
        }
        for reason_id, mapping in reason_observation.items():
            mapped_factors = set(mapping["support_factors"])
            for state_name in mapping["support_states"]:
                mapped_factors.update(state_factor_closure[state_name])
            for factor_name in mapped_factors:
                if factor_name not in self.factor_index:
                    raise ValueError(f"reason {reason_id} references unknown factor {factor_name}")
                reason_factor_map[reason_id, self.factor_index[factor_name]] = True
        if not reason_factor_map.any(dim=1).all():
            raise ValueError("every reason requires at least one factor-constrained visual mask")
        self.register_buffer("reason_factor_map", reason_factor_map, persistent=True)

        self.reason_queries = nn.Parameter(torch.randn(21, dim) * 0.02)
        self.state_embeddings = nn.Parameter(torch.randn(len(self.state_names), dim) * 0.02)
        self.semantic_attention = nn.MultiheadAttention(dim, self_attention_heads, batch_first=True)
        self.semantic_norm = nn.LayerNorm(dim)
        self.visual_decoder = MOSAICSparseLabelDecoder(
            21,
            dim=dim,
            decoder_layers=decoder_layers,
            self_attention_heads=self_attention_heads,
            highres_topk=highres_topk,
            midres_topk=midres_topk,
        )
        self.classifier_weight = nn.Parameter(torch.empty(21, dim * 2))
        self.classifier_bias = nn.Parameter(torch.zeros(21))
        nn.init.xavier_uniform_(self.classifier_weight)

    def _positive_state_factors(
        self,
        state_name: str,
        states: dict[str, Any],
        visiting: set[str],
    ) -> set[str]:
        if state_name in visiting:
            raise ValueError(f"state dependency cycle detected at {state_name}")
        visiting = set(visiting)
        visiting.add(state_name)
        factors: set[str] = set()
        for group in states[state_name]["required_groups"]:
            for reference in group["any_of"]:
                if reference in states:
                    factors.update(self._positive_state_factors(reference, states, visiting))
                else:
                    factors.add(reference)
        return factors

    def _reason_masks(self, factor_masks: torch.Tensor) -> torch.Tensor:
        clamped = factor_masks.clamp(0.0, 1.0 - 1e-6)
        log_absence = torch.log1p(-clamped)
        union_log_absence = torch.einsum(
            "rf,bfhw->brhw", self.reason_factor_map.to(dtype=log_absence.dtype), log_absence
        )
        return 1.0 - union_log_absence.exp()

    def forward(
        self,
        pyramid: dict[str, torch.Tensor],
        factor_features: torch.Tensor,
        factor_soft_masks: torch.Tensor,
        state_prob: torch.Tensor,
        state_uncertainty: torch.Tensor,
        *,
        state_contribution_cap: float = 0.20,
    ) -> dict[str, torch.Tensor]:
        if type(state_contribution_cap) not in {int, float} or not 0.0 <= float(state_contribution_cap) <= 0.20:
            raise ValueError("reason state contribution cap must be in [0,0.20]")
        batch_size = factor_features.shape[0]
        if tuple(factor_features.shape) != (batch_size, len(self.factor_names), self.dim):
            raise ValueError("reason decoder factor feature shape contract is invalid")
        if tuple(factor_soft_masks.shape) != (batch_size, len(self.factor_names), 45, 80):
            raise ValueError("reason decoder factor mask shape contract is invalid")
        state_shape = (batch_size, len(self.state_names))
        if tuple(state_prob.shape) != state_shape or tuple(state_uncertainty.shape) != state_shape:
            raise ValueError("reason decoder state shape contract is invalid")

        state_confidence = (
            float(state_contribution_cap)
            * state_prob
            * (1.0 - state_uncertainty.clamp(0.0, 1.0))
        )
        state_tokens = self.state_embeddings.unsqueeze(0) * state_confidence.unsqueeze(-1)
        semantic_tokens = torch.cat((factor_features, state_tokens), dim=1)
        queries = self.reason_queries.unsqueeze(0).expand(batch_size, -1, -1)
        semantic_nodes, semantic_attention = self.semantic_attention(
            queries,
            semantic_tokens,
            semantic_tokens,
            need_weights=True,
            average_attn_weights=True,
        )
        semantic_nodes = self.semantic_norm(queries + semantic_nodes)
        reason_masks = self._reason_masks(factor_soft_masks)
        visual = self.visual_decoder(pyramid, query_seed=semantic_nodes, highres_masks=reason_masks)
        visual_nodes = visual["label_nodes"]
        joined = torch.cat((semantic_nodes, visual_nodes), dim=-1)
        logits = torch.einsum("brd,rd->br", joined, self.classifier_weight) + self.classifier_bias
        return {
            "reason_logits_latent": logits,
            "reason_nodes_semantic": semantic_nodes,
            "reason_nodes_visual": visual_nodes,
            "reason_factor_masks": reason_masks,
            "reason_semantic_attention": semantic_attention,
        }
