from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .acpr_sparse_ops import entmax15_bisect
from .mosaic_geometry_typed_attention import MOSAICGeometryTypedAttention
from .mosaic_typed_evidence_splat import typed_evidence_splat


class _FactorwiseLinear(nn.Module):
    def __init__(self, factor_count: int, dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(factor_count, dim))
        self.bias = nn.Parameter(torch.zeros(factor_count))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, factor_features: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bfd,fd->bf", factor_features, self.weight) + self.bias


class MOSAICMultiPrototypeFactorBank(nn.Module):
    _PRIOR_NAMES = {"upper_front", "front_center", "left_corridor", "right_corridor", "center_corridor"}

    def __init__(
        self,
        num_prototypes: Sequence[int],
        region_priors: Sequence[str],
        *,
        dim: int = 384,
        prior_scale_init: float = 0.05,
        prior_scale_max: float = 0.20,
        prior_dropout: float = 0.50,
        content_temperature_init: float = 0.07,
    ) -> None:
        super().__init__()
        num_prototypes = tuple(num_prototypes)
        region_priors = tuple(region_priors)
        if not num_prototypes or len(num_prototypes) != len(region_priors):
            raise ValueError("prototype counts and region priors must have equal nonzero length")
        if any(type(count) is not int or count < 2 or count > 4 for count in num_prototypes):
            raise ValueError("each MOSAIC factor requires 2..4 independent prototypes")
        if any(prior not in self._PRIOR_NAMES for prior in region_priors):
            raise ValueError("factor uses an unsupported region prior")
        if not 0.0 <= prior_dropout < 1.0:
            raise ValueError("prior_dropout must be in [0,1)")
        if not 0.0 <= prior_scale_init <= prior_scale_max <= 0.20:
            raise ValueError("spatial prior scales must satisfy 0 <= init <= max <= 0.20")
        if not 0.02 < content_temperature_init < 0.50:
            raise ValueError("content temperature must be in (0.02,0.50)")

        self.factor_count = len(num_prototypes)
        self.dim = dim
        self.max_prototypes = max(num_prototypes)
        self.prior_scale_max = float(prior_scale_max)
        self.prior_dropout = float(prior_dropout)
        valid_mask = torch.arange(self.max_prototypes).unsqueeze(0) < torch.tensor(num_prototypes).unsqueeze(1)
        self.register_buffer("prototype_valid_mask", valid_mask, persistent=True)
        self.prototypes = nn.Parameter(torch.randn(self.factor_count, self.max_prototypes, dim) * 0.02)
        self.key_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.context_router = nn.Linear(dim, self.factor_count * self.max_prototypes)
        nn.init.zeros_(self.context_router.bias)

        temperature_fraction = (content_temperature_init - 0.02) / 0.48
        temperature_logit = math.log(temperature_fraction / (1.0 - temperature_fraction))
        self.content_temperature_raw = nn.Parameter(torch.full((self.factor_count,), temperature_logit))
        if prior_scale_max == 0:
            prior_fraction = 0.0
            prior_raw = -20.0
        else:
            prior_fraction = min(max(prior_scale_init / prior_scale_max, 1e-6), 1.0 - 1e-6)
            prior_raw = math.log(prior_fraction / (1.0 - prior_fraction))
        self.prior_scale_raw = nn.Parameter(torch.full((self.factor_count,), prior_raw))
        self.register_buffer("region_prior_maps", self._build_region_prior_maps(region_priors), persistent=True)

    @staticmethod
    def _build_region_prior_maps(region_priors: Sequence[str]) -> torch.Tensor:
        y, x = torch.meshgrid(torch.linspace(-1.0, 1.0, 12), torch.linspace(-1.0, 1.0, 20), indexing="ij")
        specifications = {
            "upper_front": (0.0, -0.65, 0.55, 0.32),
            "front_center": (0.0, 0.25, 0.45, 0.55),
            "left_corridor": (-0.52, 0.38, 0.38, 0.62),
            "right_corridor": (0.52, 0.38, 0.38, 0.62),
            "center_corridor": (0.0, 0.52, 0.38, 0.58),
        }
        maps = []
        for name in region_priors:
            center_x, center_y, scale_x, scale_y = specifications[name]
            prior = torch.exp(-0.5 * (((x - center_x) / scale_x) ** 2 + ((y - center_y) / scale_y) ** 2))
            maps.append(prior / prior.max().clamp_min(1e-6))
        return torch.stack(maps)

    @property
    def content_temperature(self) -> torch.Tensor:
        return 0.02 + 0.48 * torch.sigmoid(self.content_temperature_raw)

    @property
    def prior_scale(self) -> torch.Tensor:
        return self.prior_scale_max * torch.sigmoid(self.prior_scale_raw)

    @staticmethod
    def aggregate_prototype_scores(
        prototype_scores: torch.Tensor,
        prototype_weights: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_mask.view(1, *valid_mask.shape, 1, 1)
        log_weights = prototype_weights.clamp_min(1e-30).log()[..., None, None]
        weighted_scores = torch.where(valid, prototype_scores + log_weights, torch.full_like(prototype_scores, -torch.inf))
        return torch.logsumexp(weighted_scores, dim=2)

    def _prototype_weights(
        self,
        context: torch.Tensor,
        prior_mode: str,
        prototype_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = prototype_valid_mask.unsqueeze(0)
        if prior_mode == "prior_only":
            weights = valid.expand(context.shape[0], -1, -1).to(dtype=context.dtype)
            return weights / weights.sum(-1, keepdim=True)
        pooled = context.mean(dim=(-2, -1))
        router_logits = self.context_router(pooled).reshape(
            context.shape[0], self.factor_count, self.max_prototypes
        )
        router_logits = router_logits.masked_fill(~valid, -1e4)
        weights = entmax15_bisect(router_logits, dim=-1) * valid.to(dtype=router_logits.dtype)
        return weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)

    def _prototype_diagnostics(
        self,
        weights: torch.Tensor,
        prototypes: torch.Tensor,
        prototype_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        occupancy = weights.mean(dim=0)
        entropy = -(weights.clamp_min(1e-12).log() * weights).sum(-1).mean(0)
        normalized_prototypes = F.normalize(prototypes, dim=-1, eps=1e-6)
        pairwise = torch.einsum("fkd,fjd->fkj", normalized_prototypes, normalized_prototypes)
        valid_pairs = prototype_valid_mask[:, :, None] & prototype_valid_mask[:, None, :]
        pairwise = torch.where(valid_pairs, pairwise, torch.zeros_like(pairwise))
        return {
            "prototype_occupancy": occupancy.detach(),
            "prototype_effective_count": entropy.exp().detach(),
            "prototype_pairwise_cosine": pairwise.detach(),
            "dominant_prototype_rate": (weights.max(-1).values > 0.85).float().mean(0).detach(),
            "dead_prototype_count": ((occupancy < 1e-4) & prototype_valid_mask).sum(-1).detach(),
        }

    def forward(
        self,
        context: torch.Tensor,
        *,
        prior_mode: str = "full",
        query_permutation: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if prior_mode not in {"full", "content_only", "prior_only"}:
            raise ValueError("prior_mode must be full, content_only, or prior_only")
        if context.ndim != 4 or tuple(context.shape[1:]) != (self.dim, 12, 20):
            raise ValueError("prototype bank expects [B,D,12,20]")
        if not context.is_floating_point():
            raise ValueError("prototype bank requires floating-point context")

        if query_permutation is None:
            query_index = torch.arange(self.factor_count, device=context.device)
        else:
            query_index = query_permutation.to(device=context.device, dtype=torch.long)
            if query_index.shape != (self.factor_count,) or not torch.equal(
                torch.sort(query_index).values,
                torch.arange(self.factor_count, device=context.device),
            ):
                raise ValueError("query_permutation must be a factor permutation")
        # Shuffle only query/content semantics. The output factor keeps its
        # spatial prior, so this is an actual semantic intervention rather
        # than a post-hoc relabeling of predictions.
        prototypes = self.prototypes.index_select(0, query_index)
        valid_mask = self.prototype_valid_mask.index_select(0, query_index)
        temperatures = self.content_temperature.index_select(0, query_index)
        compute_context = context.to(dtype=prototypes.dtype)
        weights = self._prototype_weights(compute_context, prior_mode, valid_mask)
        if prior_mode == "prior_only":
            content_scores = compute_context.new_zeros(
                compute_context.shape[0], self.factor_count, self.max_prototypes, 12, 20
            )
        else:
            keys = F.normalize(self.key_proj(compute_context), dim=1, eps=1e-6)
            normalized_prototypes = F.normalize(prototypes, dim=-1, eps=1e-6)
            content_scores = torch.einsum("fkd,bdhw->bfkhw", normalized_prototypes, keys)
            content_scores = content_scores / temperatures.view(1, -1, 1, 1, 1)

        if prior_mode == "content_only":
            prior = compute_context.new_zeros(compute_context.shape[0], self.factor_count, 1, 12, 20)
        else:
            prior = self.prior_scale.view(1, -1, 1, 1, 1) * self.region_prior_maps.view(
                1, self.factor_count, 1, 12, 20
            )
            if self.training and prior_mode == "full" and self.prior_dropout > 0:
                keep = torch.rand(
                    compute_context.shape[0], self.factor_count, 1, 1, 1, device=compute_context.device
                ) >= self.prior_dropout
                prior = prior * keep.to(dtype=prior.dtype)

        scores = content_scores + prior
        visible_scores = torch.where(
            valid_mask.view(1, self.factor_count, self.max_prototypes, 1, 1),
            scores,
            torch.zeros_like(scores),
        )
        coarse_scores = self.aggregate_prototype_scores(scores, weights, valid_mask)
        return {
            "prototype_scores": visible_scores,
            "prototype_weights": weights,
            "coarse_scores": coarse_scores,
            "prior_scale": self.prior_scale,
            "prototype_stats": self._prototype_diagnostics(weights, prototypes, valid_mask),
            "prototype_queries": prototypes,
            "prototype_valid_mask": valid_mask,
        }


class MOSAICObservablePredicateLayer(nn.Module):
    def __init__(
        self,
        factors: Sequence[dict[str, Any]],
        *,
        dim: int = 384,
        anchors_per_factor: int = 2,
        heads: int = 4,
        point_samples: int = 4,
        curve_samples: int = 16,
        region_samples: int = 12,
        prior_scale_init: float = 0.05,
        prior_scale_max: float = 0.20,
        prior_dropout: float = 0.50,
        content_temperature_init: float = 0.07,
    ) -> None:
        super().__init__()
        factors = tuple(factors)
        if not factors:
            raise ValueError("observable predicate layer requires factor definitions")
        required = {"name", "type", "num_prototypes"}
        if any(not isinstance(factor, dict) or not required <= set(factor) for factor in factors):
            raise ValueError("observable predicate factor definitions are incomplete")
        region_priors: list[str] = []
        for factor in factors:
            if "weak_regions" in factor:
                weak_regions = factor["weak_regions"]
                if not isinstance(weak_regions, list) or len(weak_regions) != 1 or not isinstance(weak_regions[0], str):
                    raise ValueError("IC-DOR factors must provide exactly one observable weak region")
                region_priors.append(weak_regions[0])
            elif "region_prior" in factor:
                # Legacy MOSAIC-AD compatibility; IC-DOR config uses weak_regions.
                region_priors.append(str(factor["region_prior"]))
            else:
                raise ValueError("observable predicate factor definitions require weak_regions")
        self.factor_names = tuple(str(factor["name"]) for factor in factors)
        self.factor_types = tuple(str(factor["type"]) for factor in factors)
        self.factor_count = len(factors)
        self.dim = dim
        self.anchors_per_factor = anchors_per_factor
        self.prototype_bank = MOSAICMultiPrototypeFactorBank(
            tuple(int(factor["num_prototypes"]) for factor in factors),
            tuple(region_priors),
            dim=dim,
            prior_scale_init=prior_scale_init,
            prior_scale_max=prior_scale_max,
            prior_dropout=prior_dropout,
            content_temperature_init=content_temperature_init,
        )
        self.typed_attention = MOSAICGeometryTypedAttention(
            self.factor_types,
            dim=dim,
            anchors_per_factor=anchors_per_factor,
            heads=heads,
            point_samples=point_samples,
            curve_samples=curve_samples,
            region_samples=region_samples,
        )
        anchor_fraction = (2.0 - 0.5) / 2.5
        anchor_logit = math.log(anchor_fraction / (1.0 - anchor_fraction))
        self.anchor_temperature_raw = nn.Parameter(torch.full((self.factor_count,), anchor_logit))
        self.sample_key_proj = nn.Linear(dim, dim, bias=False)
        self.sample_value_proj = nn.Linear(dim, dim, bias=False)
        self.mid_key_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.mid_value_proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.feature_fusion = nn.LayerNorm(dim)
        self.presence_head = _FactorwiseLinear(self.factor_count, dim)
        self.visibility_head = _FactorwiseLinear(self.factor_count, dim)
        y, x = torch.meshgrid(torch.linspace(-1.0, 1.0, 12), torch.linspace(-1.0, 1.0, 20), indexing="ij")
        self.register_buffer("context_coordinate_grid", torch.stack((x.reshape(-1), y.reshape(-1)), dim=-1))

    @property
    def anchor_temperature(self) -> torch.Tensor:
        return 0.5 + 2.5 * torch.sigmoid(self.anchor_temperature_raw)

    def _standardized_anchor_logits(self, coarse_scores: torch.Tensor) -> torch.Tensor:
        flat = coarse_scores.flatten(-2)
        mean = flat.mean(-1, keepdim=True)
        scale = flat.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        return (flat - mean) / scale / self.anchor_temperature.view(1, -1, 1)

    def _soft_anchors(self, coarse_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self._standardized_anchor_logits(coarse_scores)
        first_distribution = entmax15_bisect(logits, dim=-1)
        first_anchor = torch.einsum("bfn,nc->bfc", first_distribution, self.context_coordinate_grid)
        displacement = self.context_coordinate_grid.view(1, 1, -1, 2) - first_anchor.unsqueeze(-2)
        inhibition = 3.0 * torch.exp(-displacement.square().sum(-1) / (2.0 * 0.18**2))
        second_distribution = entmax15_bisect(logits - inhibition, dim=-1)
        second_anchor = torch.einsum("bfn,nc->bfc", second_distribution, self.context_coordinate_grid)
        anchors = torch.stack((first_anchor, second_anchor), dim=2)
        mask_distribution = entmax15_bisect(logits, dim=-1)
        return anchors, mask_distribution

    def _read_sparse_samples(
        self,
        sampled_features: torch.Tensor,
        prototype_weights: torch.Tensor,
        prototype_queries: torch.Tensor,
        prototype_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sampled_features = sampled_features.to(dtype=self.sample_key_proj.weight.dtype)
        sample_keys = F.normalize(self.sample_key_proj(sampled_features), dim=-1, eps=1e-6)
        normalized_queries = F.normalize(prototype_queries, dim=-1, eps=1e-6)
        sample_logits = torch.einsum("bfmhsd,fkd->bfkmhs", sample_keys, normalized_queries)
        sample_valid = self.typed_attention.sample_valid_mask.view(1, self.factor_count, 1, 1, 1, -1)
        prototype_valid = prototype_valid_mask.view(
            1, self.factor_count, self.prototype_bank.max_prototypes, 1, 1, 1
        )
        sample_logits = sample_logits.masked_fill(~(sample_valid & prototype_valid), -1e4)
        flat_attention = entmax15_bisect(sample_logits.flatten(3), dim=-1)
        values = self.sample_value_proj(sampled_features).flatten(2, 4)
        prototype_features = torch.einsum("bfkt,bftd->bfkd", flat_attention, values)
        factor_features = torch.einsum("bfk,bfkd->bfd", prototype_weights, prototype_features)
        prototype_support = (flat_attention > 1e-5).float().sum(-1)
        # The typed splat/rereader needs one attention distribution over the
        # actual typed samples. Keep prototype diagnostics separate, and
        # marginalize the prototype axis with the learned prototype weights
        # before reshaping to [B,F,anchors,heads,samples].
        sample_attention = torch.einsum("bfk,bfkt->bft", prototype_weights, flat_attention)
        return factor_features, sample_attention, prototype_support

    def _factor_queries(
        self,
        prototype_weights: torch.Tensor,
        prototype_queries: torch.Tensor,
        prototype_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        prototype_mask = prototype_valid_mask.to(dtype=prototype_queries.dtype)
        return torch.einsum(
            "bfk,fkd->bfd",
            prototype_weights.to(dtype=prototype_queries.dtype),
            prototype_queries * prototype_mask.unsqueeze(-1),
        )

    def _read_mid_features(
        self,
        middle: torch.Tensor,
        prototype_weights: torch.Tensor,
        prototype_queries: torch.Tensor,
        prototype_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        middle = middle.to(dtype=self.mid_key_proj.weight.dtype)
        keys = F.normalize(self.mid_key_proj(middle).flatten(2).transpose(1, 2), dim=-1, eps=1e-6)
        queries = F.normalize(prototype_queries, dim=-1, eps=1e-6)
        scores = torch.einsum("fkd,bnd->bfkn", queries, keys)
        scores = scores.masked_fill(~prototype_valid_mask.view(1, self.factor_count, -1, 1), -1e4)
        attention = entmax15_bisect(scores, dim=-1)
        values = self.mid_value_proj(middle).flatten(2).transpose(1, 2)
        prototype_features = torch.einsum("bfkn,bnd->bfkd", attention, values)
        return torch.einsum("bfk,bfkd->bfd", prototype_weights, prototype_features), (attention > 1e-5).float().sum(-1)

    @staticmethod
    def _binary_entropy(probability: torch.Tensor) -> torch.Tensor:
        probability = probability.clamp(1e-6, 1.0 - 1e-6)
        return -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log()) / math.log(2.0)

    def forward(
        self,
        pyramid: dict[str, torch.Tensor],
        *,
        prior_mode: str = "full",
        query_permutation: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if not isinstance(pyramid, dict) or not {"F_hi", "F_mid", "F_ctx"} <= set(pyramid):
            raise ValueError("observable predicate layer requires F_hi/F_mid/F_ctx")
        high, middle, context = pyramid["F_hi"], pyramid["F_mid"], pyramid["F_ctx"]
        batch_size = high.shape[0] if high.ndim else -1
        if (
            tuple(high.shape) != (batch_size, self.dim, 45, 80)
            or tuple(middle.shape) != (batch_size, self.dim, 23, 40)
            or tuple(context.shape) != (batch_size, self.dim, 12, 20)
        ):
            raise ValueError("observable predicate pyramid has invalid scale shapes")

        prototype_output = self.prototype_bank(
            context,
            prior_mode=prior_mode,
            query_permutation=query_permutation,
        )
        anchors, coarse_distribution = self._soft_anchors(prototype_output["coarse_scores"])
        sampling_high = torch.zeros_like(high) if prior_mode == "prior_only" else high
        typed_output = self.typed_attention(sampling_high, anchors)
        factor_queries = self._factor_queries(
            prototype_output["prototype_weights"],
            prototype_output["prototype_queries"],
            prototype_output["prototype_valid_mask"],
        )
        if prior_mode == "prior_only":
            factor_features = factor_queries
            sample_attention = factor_queries.new_zeros(
                batch_size,
                self.factor_count,
                self.typed_attention.anchors_per_factor
                * self.typed_attention.heads
                * self.typed_attention.max_samples,
            )
            sparse_prototype_support = factor_queries.new_zeros(
                batch_size, self.factor_count, self.prototype_bank.max_prototypes
            )
            mid_prototype_support = sparse_prototype_support.clone()
        else:
            sparse_features, sample_attention, sparse_prototype_support = self._read_sparse_samples(
                typed_output["sampled_features"],
                prototype_output["prototype_weights"],
                prototype_output["prototype_queries"],
                prototype_output["prototype_valid_mask"],
            )
            mid_features, mid_prototype_support = self._read_mid_features(
                middle,
                prototype_output["prototype_weights"],
                prototype_output["prototype_queries"],
                prototype_output["prototype_valid_mask"],
            )
            factor_features = self.feature_fusion(sparse_features + mid_features)
        expected_sample_count = (
            self.typed_attention.anchors_per_factor
            * self.typed_attention.heads
            * self.typed_attention.max_samples
        )
        if sample_attention.shape != (batch_size, self.factor_count, expected_sample_count):
            raise RuntimeError(
                "observable predicate sample attention must be prototype-marginalized "
                f"to [B,F,{expected_sample_count}], got {tuple(sample_attention.shape)}"
            )
        sample_attention_typed = sample_attention.reshape(
            batch_size,
            self.factor_count,
            self.typed_attention.anchors_per_factor,
            self.typed_attention.heads,
            self.typed_attention.max_samples,
        )
        presence_logits = self.presence_head(factor_features)
        visibility_logits = self.visibility_head(factor_features)
        presence_probability = torch.sigmoid(presence_logits)
        visibility_probability = torch.sigmoid(visibility_logits)
        positive_evidence = visibility_probability * presence_probability
        negative_evidence = visibility_probability * (1.0 - presence_probability)
        uncertainty = 0.5 * (
            self._binary_entropy(presence_probability) + self._binary_entropy(visibility_probability)
        )
        coarse_masks = coarse_distribution.reshape(batch_size, self.factor_count, 12, 20)
        coarse_upsample = F.interpolate(coarse_masks, size=(45, 80), mode="bilinear", align_corners=False)
        coarse_upsample = coarse_upsample / coarse_upsample.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        splat = typed_evidence_splat(
            typed_output["sampling_coordinates"], typed_output["sampled_features"], sample_attention_typed,
            self.factor_types, output_hw=(45, 80), coarse_hw=(12, 20),
        )
        soft_masks = splat["fine_mask"]
        anchor_separation = (anchors[:, :, 0] - anchors[:, :, 1]).norm(dim=-1)
        measurement_stats = {
            **prototype_output["prototype_stats"],
            "anchor_separation_mean": anchor_separation.mean().detach(),
            "sample_attention_support_mean": (sample_attention > 1e-5).float().sum(-1).mean().detach(),
            "fine_prototype_support": sparse_prototype_support.detach(),
            "mid_prototype_support": mid_prototype_support.detach(),
            "fine_mask_delta_mean": (soft_masks - coarse_upsample).abs().mean().detach(),
            "fine_mask_delta_max": (soft_masks - coarse_upsample).abs().amax().detach(),
        }
        return {
            "factor_features": factor_features,
            "factor_presence_logits": presence_logits,
            "factor_presence_prob": presence_probability,
            "factor_visibility_logits": visibility_logits,
            "factor_visibility_prob": visibility_probability,
            "factor_positive_evidence": positive_evidence,
            "factor_negative_evidence": negative_evidence,
            "factor_uncertainty": uncertainty,
            "factor_soft_masks": soft_masks,
            "factor_coarse_masks": coarse_upsample,
            "factor_fine_features": splat["fine_features"],
            "prototype_weights": prototype_output["prototype_weights"],
            "prototype_scores": prototype_output["prototype_scores"],
            "anchor_coordinates": anchors,
            "sampling_coordinates": typed_output["sampling_coordinates"],
            "sampled_features": typed_output["sampled_features"],
            "sample_attention": sample_attention_typed,
            "sample_valid_mask": typed_output["sample_valid_mask"],
            "prior_scale": prototype_output["prior_scale"],
            "measurement_stats": measurement_stats,
        }
