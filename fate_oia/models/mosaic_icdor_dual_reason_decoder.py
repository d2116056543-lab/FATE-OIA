from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .mosaic_reason_policy import bounded_reason_residual

from .acpr_sparse_ops import entmax15_bisect
from .mosaic_factor_seeded_rereader import MOSAICFactorSeededRereader
from .mosaic_sparse_label_decoder import MOSAICSparseLabelDecoder


class MOSAICICDORVisualReasonDecoder(nn.Module):
    """Direct visual observed-reason path, independent of factors and states."""

    def __init__(
        self,
        *,
        dim: int = 384,
        decoder_layers: int = 2,
        self_attention_heads: int = 4,
        highres_topk: int = 256,
        midres_topk: int = 128,
    ) -> None:
        super().__init__()
        self.decoder = MOSAICSparseLabelDecoder(
            21,
            dim=dim,
            decoder_layers=decoder_layers,
            self_attention_heads=self_attention_heads,
            highres_topk=highres_topk,
            midres_topk=midres_topk,
        )

    def forward(self, reason_pyramid: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        output = self.decoder(reason_pyramid)
        return {
            "reason_visual_observed_logits": output["label_logits"],
            "reason_visual_observed_nodes": output["label_nodes"],
            "reason_visual_observed_attention": output["retrieval_attention"],
        }


class MOSAICICDORLatentReasonDecoder(nn.Module):
    """Hard factor-allowed latent reason path with explicit escape tokens."""

    def __init__(
        self,
        ontology: dict[str, Any],
        *,
        dim: int = 384,
        decoder_layers: int = 2,
        self_attention_heads: int = 4,
        highres_topk: int = 256,
        midres_topk: int = 128,
    ) -> None:
        super().__init__()
        factor_count = len(ontology["factors"])
        factor_index = ontology["factor_index"]
        routes = ontology["reason_routes"]
        allow = torch.zeros(21, factor_count, dtype=torch.bool)
        absence = torch.zeros(21, factor_count, dtype=torch.bool)
        for reason_index, route in routes.items():
            for factor_name in route["latent_factors"]:
                allow[reason_index, factor_index[factor_name]] = True
            for factor_name in route.get("absence_factors", []):
                absence[reason_index, factor_index[factor_name]] = True
        if not allow.any(dim=-1).all():
            raise ValueError("IC-DOR latent reason decoder requires hard factors for every reason")
        self.register_buffer("reason_factor_allow_mask", allow, persistent=True)
        self.register_buffer("reason_factor_absence_mask", absence, persistent=True)
        self.reason_queries = nn.Parameter(torch.randn(21, dim) * 0.02)
        self.escape_tokens = nn.Parameter(torch.randn(21, dim) * 0.02)
        self.factor_key = nn.Linear(dim, dim, bias=False)
        self.reason_query = nn.Linear(dim, dim, bias=False)
        self.semantic_norm = nn.LayerNorm(dim)
        # This is target-owned rereading of the factor layer's typed samples.
        # Its inputs are detached, preserving the CREDO firewall from latent
        # reason supervision back into visual factor measurement.
        self.typed_rereader = MOSAICFactorSeededRereader(dim=dim, target_count=21)
        self.typed_transport_gain = nn.Parameter(torch.tensor(0.10))
        self.decoder = MOSAICSparseLabelDecoder(
            21,
            dim=dim,
            decoder_layers=decoder_layers,
            self_attention_heads=self_attention_heads,
            highres_topk=highres_topk,
            midres_topk=midres_topk,
            mask_fallback_floor=1e-8,
        )
        self.classifier = nn.Linear(2 * dim, 1, bias=False)

    def _masked_visual_nodes(
        self,
        reason_pyramid: dict[str, torch.Tensor],
        semantic: torch.Tensor,
        reason_factor_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Verify each latent reason without cross-reason semantic leakage."""
        decoder = self.decoder
        high = reason_pyramid["F_hi"]
        middle = reason_pyramid["F_mid"]
        context = reason_pyramid["F_ctx"]
        batch_size = high.shape[0]
        queries = decoder.label_queries.unsqueeze(0).expand(batch_size, -1, -1) + semantic
        context_tokens = context.flatten(2).transpose(1, 2)
        context_nodes, _ = decoder.context_attention(queries, context_tokens, context_tokens, need_weights=False)
        nodes = decoder.context_norm(queries + context_nodes)

        high_keys = F.normalize(decoder.high_key(high).flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        mid_keys = F.normalize(decoder.mid_key(middle).flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        normalized_nodes = F.normalize(nodes, dim=-1, eps=1e-6)
        high_scores = torch.einsum("bld,bnd->bln", normalized_nodes, high_keys) / math.sqrt(decoder.dim)
        mid_scores = torch.einsum("bld,bnd->bln", normalized_nodes, mid_keys) / math.sqrt(decoder.dim)
        high_scores = decoder._masked_scores(high_scores, reason_factor_masks, (45, 80))
        mid_scores = decoder._masked_scores(mid_scores, reason_factor_masks, (23, 40))
        high_values, high_indices = high_scores.topk(decoder.highres_topk, dim=-1)
        mid_values, mid_indices = mid_scores.topk(decoder.midres_topk, dim=-1)
        high_tokens = decoder.high_value(high).flatten(2).transpose(1, 2)
        mid_tokens = decoder.mid_value(middle).flatten(2).transpose(1, 2)
        gathered = torch.cat((
            decoder._gather_tokens(high_tokens, high_indices),
            decoder._gather_tokens(mid_tokens, mid_indices),
        ), dim=2)
        retrieval_attention = entmax15_bisect(torch.cat((high_values, mid_values), dim=-1), dim=-1)
        nodes = decoder.retrieval_norm(nodes + torch.einsum("blk,blkd->bld", retrieval_attention, gathered))

        reason_count = nodes.shape[1]
        cross_reason_mask = torch.ones(reason_count, reason_count, dtype=torch.bool, device=nodes.device)
        cross_reason_mask.fill_diagonal_(False)
        for block in decoder.blocks:
            attended, _ = block.attention(nodes, nodes, nodes, need_weights=False, attn_mask=cross_reason_mask)
            nodes = block.norm_attention(nodes + attended)
            nodes = block.norm_feed_forward(nodes + block.feed_forward(nodes))
        return decoder.final_norm(nodes), retrieval_attention

    def forward(
        self,
        reason_pyramid: dict[str, torch.Tensor],
        factor_features: torch.Tensor,
        factor_soft_masks: torch.Tensor,
        factor_route_enabled: torch.Tensor,
        factor_positive_evidence: torch.Tensor | None = None,
        factor_negative_evidence: torch.Tensor | None = None,
        sampling_coordinates: torch.Tensor | None = None,
        sampled_features: torch.Tensor | None = None,
        sample_attention: torch.Tensor | None = None,
        factor_semantic_compatibility: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, factor_count, dim = factor_features.shape
        if tuple(factor_soft_masks.shape) != (batch_size, factor_count, 45, 80):
            raise ValueError("IC-DOR latent reason decoder factor masks must be [B,F,45,80]")
        if factor_route_enabled.shape != (factor_count,) or factor_route_enabled.dtype != torch.bool:
            raise ValueError("IC-DOR latent reason decoder needs a [F] boolean certificate route mask")
        route_features = factor_features.detach()
        if factor_positive_evidence is not None and factor_negative_evidence is not None:
            if factor_positive_evidence.shape != (batch_size, factor_count) or factor_negative_evidence.shape != (batch_size, factor_count):
                raise ValueError("IC-DOR latent reason evidence must be [B,F]")
            route_evidence = torch.where(
                self.reason_factor_absence_mask.view(1, 21, factor_count),
                factor_negative_evidence[:, None, :],
                factor_positive_evidence[:, None, :],
            )
        else:
            route_evidence = factor_features.new_ones(batch_size, 21, factor_count)
        factor_keys = self.factor_key(route_features) * route_evidence.mean(1).unsqueeze(-1)
        queries = self.reason_query(self.reason_queries).unsqueeze(0).expand(batch_size, -1, -1)
        finite_scores = torch.einsum("brd,bfd->brf", queries, factor_keys) / math.sqrt(dim)
        if factor_semantic_compatibility is not None:
            if factor_semantic_compatibility.shape != (21, factor_count):
                raise ValueError("IC-DOR semantic compatibility must be [21,F]")
            # cS is a prior audit state for the reason semantic path only.
            # A floor preserves latent-route learning access before a target
            # relationship has enough intervention evidence to be trusted.
            finite_scores = finite_scores + factor_semantic_compatibility.detach().clamp_min(0.10).log().unsqueeze(0)
        allowed = self.reason_factor_allow_mask & factor_route_enabled.view(1, -1)
        # Hard-mask before entmax: disabled factors must not affect the
        # allowed-factor distribution or the escape-token weight. Keep the
        # floor relative to allowed scores so bisection remains well-scaled.
        escape_scores = finite_scores.new_zeros(batch_size, 21, 1)
        allowed_scores = finite_scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        floor = torch.maximum(allowed_scores.amax(dim=-1, keepdim=True), escape_scores) - 100.0
        masked_scores = torch.where(allowed.unsqueeze(0), finite_scores, floor)
        unconstrained = entmax15_bisect(
            torch.cat((masked_scores, escape_scores), dim=-1), dim=-1
        )
        factor_weights = torch.where(
            allowed.unsqueeze(0),
            unconstrained[:, :, :factor_count],
            torch.zeros_like(unconstrained[:, :, :factor_count]),
        )
        escape_weight = 1.0 - factor_weights.sum(dim=-1, keepdim=True)
        semantic = torch.einsum("brf,bfd->brd", factor_weights * route_evidence, route_features)
        typed_nodes = torch.zeros_like(semantic)
        if sampling_coordinates is not None or sampled_features is not None or sample_attention is not None:
            if sampling_coordinates is None or sampled_features is None or sample_attention is None:
                raise ValueError("IC-DOR typed reason transport requires coordinates, features, and attention together")
            typed = self.typed_rereader(
                reason_pyramid["F_hi"].detach(),
                queries.detach(),
                sampling_coordinates.detach(),
                sampled_features.detach(),
                sample_attention.detach(),
                factor_weights.transpose(1, 2).detach(),
            )
            typed_active = factor_weights.sum(dim=-1, keepdim=True).detach()
            typed_nodes = typed["target_nodes"] * typed_active
        typed_gain = self.typed_transport_gain.clamp(0.0, 0.25)
        semantic = self.semantic_norm(
            semantic + escape_weight * self.escape_tokens.unsqueeze(0) + typed_gain * typed_nodes
        )
        reason_factor_masks = torch.einsum("brf,bfhw->brhw", factor_weights, factor_soft_masks.detach())
        active = reason_factor_masks.flatten(2).amax(dim=-1) > 1e-8
        visual_nodes, visual_attention = self._masked_visual_nodes(reason_pyramid, semantic, reason_factor_masks)
        visual_nodes = visual_nodes * active.unsqueeze(-1).to(semantic.dtype)
        latent_logits = self.classifier(torch.cat((semantic, visual_nodes), dim=-1)).squeeze(-1)
        return {
            "reason_logits_latent": latent_logits,
            "reason_nodes_latent": semantic,
            "reason_factor_router_weights": factor_weights,
            "reason_escape_weight": escape_weight.squeeze(-1),
            "reason_factor_masks": reason_factor_masks,
            "reason_latent_visual_nodes": visual_nodes,
            "reason_latent_visual_attention": visual_attention,
            "reason_typed_transport_nodes": typed_nodes,
            "reason_typed_transport_gain": typed_gain.detach(),
            "reason_semantic_compatibility_effective": (
                torch.ones(21, factor_count, device=semantic.device, dtype=semantic.dtype)
                if factor_semantic_compatibility is None else factor_semantic_compatibility.detach()
            ),
        }


class MOSAICICDORObservedReasonMixer(nn.Module):
    """Use direct visual reason as primary with a bounded annotation residual."""

    def __init__(self, *, init_mix: float = 0.05) -> None:
        super().__init__()
        if not 0.0 < init_mix < 0.25:
            raise ValueError("IC-DOR observed-reason residual init must be in (0,0.25)")
        # The direct visual reason path remains primary. Unlike the previous
        # compatibility shim, the resolved config now controls the initial
        # bounded residual exactly, so run artifacts can reproduce it.
        initial_alpha = float(init_mix)
        raw = math.log(initial_alpha / (0.25 - initial_alpha))
        self.alpha_raw = nn.Parameter(torch.full((21,), raw))
        self.max_alpha = 0.25

    def forward(
        self,
        reason_visual_observed_logits: torch.Tensor,
        reason_observation_logits: torch.Tensor,
        *,
        latent_enabled: bool,
    ) -> dict[str, torch.Tensor]:
        if reason_visual_observed_logits.shape != reason_observation_logits.shape or reason_visual_observed_logits.shape[-1] != 21:
            raise ValueError("IC-DOR observed-reason mixer expects matching [B,21] logits")
        alpha = (self.max_alpha * torch.sigmoid(self.alpha_raw)).unsqueeze(0)
        observed = (
            bounded_reason_residual(
                reason_visual_observed_logits,
                reason_observation_logits,
                max_alpha=self.max_alpha,
                alpha=alpha.squeeze(0),
            )[0]
            if latent_enabled
            else reason_visual_observed_logits
        )
        return {"reason_observed_logits": observed, "reason_observed_mix_gate": alpha.expand_as(observed)}
