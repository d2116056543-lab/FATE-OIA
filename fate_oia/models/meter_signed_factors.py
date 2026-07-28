from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .acpr_sparse_ops import entmax15_bisect
from .meter_meta_adapters import METERFactorMetaAdapters


class METERsignedFactors(nn.Module):
    """Signed factor evidence from full DINO fields without factor-token expansion."""

    def __init__(
        self,
        dim: int = 384,
        factor_dim: int = 21,
        num_layers: int = 3,
        rank: int = 16,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.factor_dim = int(factor_dim)
        self.num_layers = int(num_layers)
        # Keep both signed queries trainable and non-zero at initialization.
        # A 0.02 scale made support/counter maps numerically indistinguishable
        # on the 384-d frozen field during the first updates.
        self.support_embedding = nn.Parameter(torch.empty(dim))
        self.counter_embedding = nn.Parameter(torch.empty(dim))
        signed_seed = torch.randn(dim) * 0.35
        with torch.no_grad():
            self.support_embedding.copy_(signed_seed)
            self.counter_embedding.copy_(-signed_seed)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.value_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.null_keys = nn.Parameter(torch.randn(2, num_layers, dim) * 0.04)
        self.null_values = nn.Parameter(torch.randn(2, num_layers, dim) * 0.04)
        self.null_logit_offset = nn.Parameter(
            torch.full((2, factor_dim, num_layers), math.log(0.10 / 0.90))
        )
        self.layer_delta = nn.Parameter(torch.zeros(factor_dim, num_layers))
        self.evidence_proj = nn.Linear(dim, dim, bias=False)
        self.evidence_norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.zeros(factor_dim))
        self.support_score_head = nn.Linear(dim, 1)
        self.counter_score_head = nn.Linear(dim, 1)
        self.meta_adapters = METERFactorMetaAdapters(factor_dim=factor_dim, dim=dim, rank=rank)

    @staticmethod
    def _sparse_ramp(progress: float) -> float:
        return float(min(max(progress / 0.10, 0.0), 1.0))

    def _patch_distribution(self, scores: Tensor, progress: float) -> Tensor:
        dense = torch.softmax(scores, dim=-1)
        sparse = entmax15_bisect(scores, dim=-1)
        ramp = self._sparse_ramp(progress)
        return (1.0 - ramp) * dense + ramp * sparse

    def _read_sign(
        self,
        query: Tensor,
        patches: Tensor,
        sign_index: int,
        progress: float,
        score_weight: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        maps: list[Tensor] = []
        nulls: list[Tensor] = []
        details: list[Tensor] = []
        patch_attributions: list[Tensor] = []
        null_attributions: list[Tensor] = []
        scale = math.sqrt(self.dim)
        for layer in range(self.num_layers):
            key = self.key_proj[layer](patches[:, layer])
            value = self.value_proj[layer](patches[:, layer])
            scores = torch.einsum("bfd,bnd->bfn", query, key) / scale
            null_score = torch.einsum(
                "bfd,d->bf", query, self.null_keys[sign_index, layer]
            )
            null_logit = (
                null_score
                - scores.mean(dim=-1)
                + self.null_logit_offset[sign_index, :, layer].view(1, -1)
            )
            null_mass = torch.sigmoid(null_logit)
            patch_distribution = self._patch_distribution(scores, progress)
            patch_map = (1.0 - null_mass).unsqueeze(-1) * patch_distribution
            detail = torch.einsum("bfn,bnd->bfd", patch_map, value)
            detail = detail + null_mass.unsqueeze(-1) * self.null_values[sign_index, layer].view(1, 1, -1)
            value_score = torch.einsum("bnd,d->bn", value, score_weight)
            patch_attributions.append(
                patch_map * value_score.unsqueeze(1)
            )
            null_attributions.append(
                null_mass
                * torch.dot(self.null_values[sign_index, layer], score_weight)
            )
            maps.append(patch_map)
            nulls.append(null_mass)
            details.append(detail)
        return (
            torch.stack(maps, dim=2),
            torch.stack(nulls, dim=2),
            torch.stack(details, dim=2),
            torch.stack(patch_attributions, dim=2),
            torch.stack(null_attributions, dim=2),
        )

    def forward(
        self,
        factor_base_tokens: Tensor,
        patch_tokens_by_layer: Tensor,
        *,
        progress: float = 1.0,
        meta_share_weight: Tensor | None = None,
        factor_parameter_override: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        if factor_base_tokens.ndim != 3 or factor_base_tokens.shape[1:] != (self.factor_dim, self.dim):
            raise ValueError("Expected factor base tokens [B,21,D]")
        if patch_tokens_by_layer.ndim != 4 or patch_tokens_by_layer.shape[1] != self.num_layers:
            raise ValueError("Expected patch field [B,3,N,D]")
        if patch_tokens_by_layer.shape[-1] != self.dim:
            raise ValueError("Factor and patch dimensions must agree")
        positive_query = self.query_proj(factor_base_tokens + self.support_embedding.view(1, 1, -1))
        negative_query = self.query_proj(factor_base_tokens + self.counter_embedding.view(1, 1, -1))
        support_layers, support_null_layers, support_detail_layers, support_attr_layers, support_null_attr_layers = self._read_sign(
            positive_query,
            patch_tokens_by_layer,
            0,
            progress,
            self.support_score_head.weight.squeeze(0),
        )
        counter_layers, counter_null_layers, counter_detail_layers, counter_attr_layers, counter_null_attr_layers = self._read_sign(
            negative_query,
            patch_tokens_by_layer,
            1,
            progress,
            self.counter_score_head.weight.squeeze(0),
        )
        layer_logits = math.log(1.0 / self.num_layers) + 0.1 * torch.tanh(self.layer_delta)
        layer_weights = torch.softmax(layer_logits, dim=-1)
        support_map = torch.einsum("fl,bfln->bfn", layer_weights, support_layers)
        counter_map = torch.einsum("fl,bfln->bfn", layer_weights, counter_layers)
        support_null = torch.einsum("fl,bfl->bf", layer_weights, support_null_layers)
        counter_null = torch.einsum("fl,bfl->bf", layer_weights, counter_null_layers)
        support_detail = torch.einsum("fl,bfld->bfd", layer_weights, support_detail_layers)
        counter_detail = torch.einsum("fl,bfld->bfd", layer_weights, counter_detail_layers)
        support_attribution = torch.einsum(
            "fl,bfln->bfn", layer_weights, support_attr_layers
        )
        counter_attribution = torch.einsum(
            "fl,bfln->bfn", layer_weights, counter_attr_layers
        )
        support_null_attribution = torch.einsum(
            "fl,bfl->bf", layer_weights, support_null_attr_layers
        )
        counter_null_attribution = torch.einsum(
            "fl,bfl->bf", layer_weights, counter_null_attr_layers
        )
        support_score = F.softplus(self.support_score_head(support_detail).squeeze(-1))
        counter_score = F.softplus(self.counter_score_head(counter_detail).squeeze(-1))
        non_null = 1.0 - 0.5 * (support_null + counter_null)
        separation = (support_score - counter_score).abs() / (support_score + counter_score + 1e-6)
        learned_reliability = (non_null * separation).clamp(0.0, 1.0)
        early_floor = 0.20 - 0.15 * self._sparse_ramp(progress)
        reliability = early_floor + (1.0 - early_floor) * learned_reliability
        evidence_delta = self.evidence_proj(support_detail - counter_detail)
        core = self.evidence_norm(
            factor_base_tokens + float(min(max(progress / 0.10, 0.0), 1.0)) * self.gamma.view(1, -1, 1) * evidence_delta
        )
        meta_delta = self.meta_adapters(core, parameter_override=factor_parameter_override)
        action_tokens = core + meta_delta
        if meta_share_weight is None:
            meta_share_weight = torch.zeros(self.factor_dim, device=core.device, dtype=core.dtype)
        omega = meta_share_weight.to(device=core.device, dtype=core.dtype).view(1, self.factor_dim, 1).clamp(0.0, 1.0)
        reason_tokens = action_tokens.detach() + float(min(max(progress / 0.10, 0.0), 1.0)) * omega * (meta_delta - meta_delta.detach())
        uncertainty = (0.5 * (support_null + counter_null) + counter_score / (support_score + counter_score + 1e-6)).clamp(0.0, 1.0)
        return {
            "factor_base_tokens": factor_base_tokens,
            "factor_core_tokens": core,
            "factor_action_tokens": action_tokens,
            "factor_to_reason_tokens": reason_tokens,
            "factor_support_maps_by_layer": support_layers,
            "factor_counter_maps_by_layer": counter_layers,
            "factor_support_null_by_layer": support_null_layers,
            "factor_counter_null_by_layer": counter_null_layers,
            "factor_support_map": support_map,
            "factor_counter_map": counter_map,
            "factor_support_null": support_null,
            "factor_counter_null": counter_null,
            "factor_support_score": support_score,
            "factor_counter_score": counter_score,
            "factor_support_detail": support_detail,
            "factor_counter_detail": counter_detail,
            "factor_support_attribution": support_attribution,
            "factor_counter_attribution": counter_attribution,
            "factor_support_null_attribution": support_null_attribution,
            "factor_counter_null_attribution": counter_null_attribution,
            "factor_reliability": reliability,
            "factor_uncertainty": uncertainty,
            "factor_layer_weights": layer_weights,
            "factor_meta_delta": meta_delta,
        }
