from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from .acpr_reason_grammar import ACPRReasonGrammar
from .acpr_sparse_ops import entmax15_bisect


class VETRAVisualFactorTransport(nn.Module):
    """Route semantic hypotheses while transporting visual values only."""

    def __init__(self, predicate_names: list[str], grammar_path: str, dim: int = 384,
                 num_layers: int = 3, action_dim: int = 4, correction_cap: float = .20,
                 null_route_prior: float = .50) -> None:
        super().__init__()
        self.dim, self.num_layers, self.action_dim = dim, num_layers, action_dim
        self.correction_cap = float(correction_cap)
        self.null_route_prior = float(null_route_prior)
        grammar = ACPRReasonGrammar(grammar_path)
        positive, contradictory = grammar.reason_predicate_matrix(predicate_names)
        compatibility = grammar.compatible_action_matrix()
        self.register_buffer("grammar_positive_mask", torch.tensor(positive, dtype=torch.bool))
        self.register_buffer("grammar_contradictory_mask", torch.tensor(contradictory, dtype=torch.bool))
        self.register_buffer("reason_action_compatibility", compatibility.bool())
        self.layer_value_proj = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(num_layers))
        self.visual_value_proj = nn.Linear(dim, dim, bias=False)
        self.semantic_key_proj = nn.Linear(dim, dim, bias=False)
        self.action_query_proj = nn.Linear(dim, dim, bias=False)
        self.compatibility_proj = nn.Linear(dim, dim, bias=False)
        self.layer_embedding = nn.Parameter(torch.randn(num_layers, dim) * .02)
        self.role_embedding = nn.Parameter(torch.randn(2, dim) * .02)
        self.unnamed_embedding = nn.Parameter(torch.randn(dim) * .02)
        self.null_key = nn.Parameter(torch.randn(2, dim) * .02)
        self.support_query_offset = nn.Parameter(torch.randn(action_dim, dim) * .02)
        self.counter_query_offset = nn.Parameter(torch.randn(action_dim, dim) * .02)
        self.reliability_weights_raw = nn.Parameter(torch.zeros(4))
        self.transport_norm = nn.LayerNorm(dim)
        self.support_head = nn.Parameter(torch.empty(action_dim, dim))
        self.counter_head = nn.Parameter(torch.empty(action_dim, dim))
        nn.init.normal_(self.support_head, std=1e-4)
        nn.init.normal_(self.counter_head, std=1e-4)

    def _visual_values(self, patch_layers: Tensor, attention: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # [B,L,N,D] -> [B,P,L,D], preserving the identity of every DINO layer.
        projected = torch.stack(
            [self.layer_value_proj[layer](patch_layers[:, layer]) for layer in range(self.num_layers)], dim=1
        )
        values = torch.einsum("bpn,blnd->bpld", attention, projected)
        mean = values.mean(2, keepdim=True)
        layer_agreement = .5 * (1 + torch.nn.functional.cosine_similarity(values, mean.expand_as(values), dim=-1))
        entropy = -(attention.clamp_min(1e-9) * attention.clamp_min(1e-9).log()).sum(-1)
        map_concentration = 1 - entropy / math.log(attention.shape[-1])
        return self.visual_value_proj(values), layer_agreement.clamp(0, 1), map_concentration.clamp(0, 1)

    def _factor_bank(self, role: int, visual_values: Tensor, reason_nodes: Tensor,
                     predicate_tokens: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
        grammar = self.grammar_positive_mask if role == 0 else self.grammar_contradictory_mask
        edge = grammar.nonzero(as_tuple=False)
        reason_ids = edge[:, 0].repeat_interleave(self.num_layers)
        predicate_ids = edge[:, 1].repeat_interleave(self.num_layers)
        layer_ids = torch.arange(self.num_layers, device=edge.device).repeat(edge.shape[0])
        batch = visual_values.shape[0]

        named_key_source = (reason_nodes[:, reason_ids] + predicate_tokens[:, predicate_ids]
                            + self.layer_embedding[layer_ids] + self.role_embedding[role])
        named_keys = self.semantic_key_proj(named_key_source)
        named_values = visual_values[:, predicate_ids, layer_ids]

        predicate_count = visual_values.shape[1]
        unnamed_predicate = torch.arange(predicate_count, device=visual_values.device).repeat_interleave(self.num_layers)
        unnamed_layer = torch.arange(self.num_layers, device=visual_values.device).repeat(predicate_count)
        unnamed_keys = self.semantic_key_proj(
            predicate_tokens[:, unnamed_predicate] + self.layer_embedding[unnamed_layer]
            + self.role_embedding[role] + self.unnamed_embedding
        )
        unnamed_values = visual_values[:, unnamed_predicate, unnamed_layer]
        null_keys = self.null_key[role].view(1, 1, -1).expand(batch, 1, -1)
        null_values = visual_values.new_zeros(batch, 1, self.dim)
        keys = torch.cat((named_keys, unnamed_keys, null_keys), 1)
        values = torch.cat((named_values, unnamed_values, null_values), 1)

        named_allowed = self.reason_action_compatibility[reason_ids].transpose(0, 1)
        allowed = torch.cat((named_allowed,
                             torch.ones(self.action_dim, unnamed_values.shape[1] + 1,
                                        dtype=torch.bool, device=visual_values.device)), 1)
        factor_predicate = torch.cat((predicate_ids, unnamed_predicate,
                                      torch.full((1,), -1, dtype=torch.long, device=visual_values.device)))
        factor_layer = torch.cat((layer_ids, unnamed_layer,
                                  torch.full((1,), -1, dtype=torch.long, device=visual_values.device)))
        meta = {"named_count": int(named_values.shape[1]), "unnamed_count": int(unnamed_values.shape[1]),
                "null_index": int(values.shape[1] - 1), "reason_ids": reason_ids,
                "predicate_ids": factor_predicate, "layer_ids": factor_layer}
        return keys, values, allowed, factor_predicate, meta

    def _route(self, role: int, action_nodes: Tensor, visual_values: Tensor, reason_nodes: Tensor,
               predicate_tokens: Tensor, predicate_probs: Tensor, layer_agreement: Tensor,
               map_concentration: Tensor, *, semantic_shuffle: bool, visual_shuffle: bool,
               force_null_only: bool, named_factors_off: bool, unnamed_factors_off: bool,
               route_off: bool, reliability_off: bool) -> dict[str, Tensor | dict[str, Any]]:
        keys, values, allowed, factor_predicate, meta = self._factor_bank(
            role, visual_values, reason_nodes, predicate_tokens)
        if semantic_shuffle and keys.shape[1] > 1:
            keys = keys.roll(1, dims=1)
        if visual_shuffle and values.shape[0] > 1:
            values = values.roll(1, dims=0)
        offset = self.support_query_offset if role == 0 else self.counter_query_offset
        query = self.action_query_proj(action_nodes + offset[None])
        score = torch.einsum("bad,bfd->baf", query, keys) / math.sqrt(self.dim)

        pred_index = factor_predicate.clamp_min(0)
        predicate_reliability = predicate_probs[:, pred_index]
        map_reliability = map_concentration[:, pred_index]
        layer_ids = meta["layer_ids"].clamp_min(0)
        layer_reliability = layer_agreement[:, pred_index, layer_ids]
        compatibility = torch.sigmoid(
            torch.einsum("bad,bfd->baf", self.compatibility_proj(action_nodes), values) / math.sqrt(self.dim)
        )
        components = torch.stack((predicate_reliability[:, None].expand_as(compatibility),
                                  map_reliability[:, None].expand_as(compatibility),
                                  layer_reliability[:, None].expand_as(compatibility), compatibility), -1)
        weights = torch.nn.functional.softplus(self.reliability_weights_raw)
        weights = weights / weights.sum()
        reliability = torch.exp((4 * weights * components.clamp_min(1e-8).log()).sum(-1))
        reliability = torch.cat((reliability[..., :-1], torch.ones_like(reliability[..., -1:])), dim=-1)
        if reliability_off:
            reliability = torch.ones_like(reliability)
        # Reliability ranks visual factors; null receives an explicit prior instead of a free advantage.
        non_null = reliability[..., :-1]
        relative = non_null / non_null.amax(-1, keepdim=True).clamp_min(1e-8)
        routing_reliability = torch.cat(
            (relative, relative.new_full(relative.shape[:-1] + (1,), self.null_route_prior)), dim=-1)
        score = score + routing_reliability.clamp_min(1e-8).log()
        score = score.masked_fill(~allowed[None], -1e4)
        if named_factors_off:
            score[..., :meta["named_count"]] = -1e4
        if unnamed_factors_off:
            start = meta["named_count"]
            score[..., start:start + meta["unnamed_count"]] = -1e4
        if force_null_only or route_off:
            score[..., :-1] = -1e4
        route = entmax15_bisect(score, dim=-1)
        context = torch.einsum("baf,bfd->bad", route, values)
        return {"route": route, "context": context, "reliability": reliability,
                "factor_values": values, "factor_keys": keys, "meta": meta}

    def forward(self, patch_tokens_by_layer_raw: Tensor, action_nodes_primary: Tensor,
                reason_nodes_primary: Tensor, predicate_tokens: Tensor, predicate_attention: Tensor,
                predicate_probs: Tensor, predicate_layer_weights: Tensor | None = None,
                *, alpha: float = 1.0, semantic_shuffle: bool = False,
                visual_shuffle: bool = False, force_null_only: bool = False,
                named_factors_off: bool = False, unnamed_factors_off: bool = False,
                support_route_off: bool = False, counter_route_off: bool = False,
                predicate_off: bool = False, reliability_off: bool = False) -> dict[str, Tensor | dict]:
        if predicate_off:
            predicate_tokens = torch.zeros_like(predicate_tokens)
            predicate_probs = torch.ones_like(predicate_probs)
            predicate_attention = torch.full_like(predicate_attention, 1 / predicate_attention.shape[-1])
        visual_values, layer_agreement, map_concentration = self._visual_values(
            patch_tokens_by_layer_raw.detach(), predicate_attention.detach())
        common = (action_nodes_primary.detach(), visual_values, reason_nodes_primary.detach(),
                  predicate_tokens.detach(), predicate_probs.detach(), layer_agreement, map_concentration)
        support = self._route(0, *common, semantic_shuffle=semantic_shuffle,
                              visual_shuffle=visual_shuffle, force_null_only=force_null_only,
                              named_factors_off=named_factors_off, unnamed_factors_off=unnamed_factors_off,
                              route_off=support_route_off, reliability_off=reliability_off)
        counter = self._route(1, *common, semantic_shuffle=semantic_shuffle,
                              visual_shuffle=visual_shuffle, force_null_only=force_null_only,
                              named_factors_off=named_factors_off, unnamed_factors_off=unnamed_factors_off,
                              route_off=counter_route_off, reliability_off=reliability_off)
        support_context = self.transport_norm(support["context"])
        counter_context = self.transport_norm(counter["context"])
        support_score = torch.einsum("bad,ad->ba", support_context, self.support_head)
        counter_score = torch.einsum("bad,ad->ba", counter_context, self.counter_head)
        delta_unscaled = self.correction_cap * torch.tanh((support_score - counter_score) / self.correction_cap)
        delta = float(alpha) * delta_unscaled
        return {
            "vetra_action_delta_unscaled": delta_unscaled,
            "vetra_action_delta": delta,
            "support_context": support_context,
            "counter_context": counter_context,
            "support_route": support["route"],
            "counter_route": counter["route"],
            "support_reliability": support["reliability"],
            "counter_reliability": counter["reliability"],
            "support_factor_values": support["factor_values"],
            "counter_factor_values": counter["factor_values"],
            "support_meta": support["meta"],
            "counter_meta": counter["meta"],
            "map_concentration": map_concentration,
            "layer_agreement": layer_agreement,
            "predicate_layer_weights_input": predicate_layer_weights,
            "reliability_component_weights": (
                torch.nn.functional.softplus(self.reliability_weights_raw)
                / torch.nn.functional.softplus(self.reliability_weights_raw).sum()
            ),
        }
