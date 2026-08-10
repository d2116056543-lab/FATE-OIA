from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .acpr_sparse_ops import entmax15_bisect


def straight_through_topk(scores: Tensor, k: int = 2) -> Tensor:
    soft = entmax15_bisect(scores, dim=-1)
    count = min(int(k), soft.shape[-1])
    index = soft.topk(count, dim=-1).indices
    hard = torch.zeros_like(soft).scatter(-1, index, soft.gather(-1, index))
    hard = hard / hard.sum(-1, keepdim=True).clamp_min(1e-8)
    return (hard - soft).detach() + soft


class DICEAtomReconstructor(nn.Module):
    region_names = ("front_center", "left_corridor", "right_corridor", "upper_traffic_region", "bottom_drivable_region")

    def __init__(self, dim: int = 384, num_layers: int = 3, num_predicates: int = 32,
                 grid_hw: tuple[int, int] = (45, 80), predicate_strength_max: float = 0.20,
                 predicate_presence_floor: float = 0.30) -> None:
        super().__init__()
        self.dim, self.num_layers, self.grid_hw = int(dim), int(num_layers), grid_hw
        self.predicate_strength_max = float(predicate_strength_max)
        self.predicate_presence_floor = float(predicate_presence_floor)
        self.query_norm = nn.LayerNorm(dim)
        self.query_proj = nn.Linear(dim, dim)
        self.predicate_query_proj = nn.Linear(dim, dim)
        self.predicate_keys = nn.Parameter(torch.randn(num_predicates, dim) * 0.02)
        self.key_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.value_proj = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.layer_router = nn.Linear(dim, num_layers)
        self.predicate_gate = nn.Linear(dim, 1)
        self.center_norm = nn.LayerNorm(dim)

    def forward(self, post_group_evidence_token: Tensor, conditioned_patch_layers: Tensor,
                predicate_attention: Tensor, predicate_probs: Tensor,
                ego_region_masks: dict[str, Tensor]) -> dict[str, Tensor]:
        evidence = post_group_evidence_token.detach()
        field = conditioned_patch_layers.detach()
        pattn = predicate_attention.detach().clamp_min(0)
        pprob = predicate_probs.detach().clamp(0, 1)
        b, layers, patches, dim = field.shape
        if layers != self.num_layers or dim != self.dim:
            raise ValueError("DICE atom reconstructor field contract mismatch")
        query_source = self.query_norm(evidence)
        query = self.query_proj(query_source)
        layer_weights = torch.softmax(self.layer_router(evidence), -1)
        scores, values = [], []
        for layer in range(layers):
            scores.append(torch.einsum("bakd,bnd->bakn", query, self.key_proj[layer](field[:, layer])) / math.sqrt(dim))
            values.append(self.value_proj[layer](field[:, layer]))
        score_stack = torch.stack(scores, 3)
        projected_values = torch.stack(values, 1)
        visual_map = torch.softmax((layer_weights[..., None] * score_stack).sum(3), -1)

        predicate_score = (torch.einsum("bakd,pd->bakp", self.predicate_query_proj(evidence), self.predicate_keys)
                           / math.sqrt(dim) + pprob[:, None, None].clamp_min(1e-8).log())
        predicate_mixture = straight_through_topk(predicate_score, 2)
        predicate_map = torch.einsum("bakp,bpn->bakn", predicate_mixture, pattn)
        predicate_map = predicate_map / predicate_map.sum(-1, keepdim=True).clamp_min(1e-8)
        agreement_product=(visual_map*predicate_map).clamp_min(0)
        agreement=torch.where(agreement_product>0,torch.sqrt(agreement_product.clamp_min(1e-12)),
                              torch.zeros_like(agreement_product)).sum(-1).clamp(0,1)
        confidence = torch.einsum("bakp,bp->bak", predicate_mixture, pprob).clamp(0, 1)
        strength = self.predicate_strength_max * agreement * confidence * torch.sigmoid(self.predicate_gate(evidence)).squeeze(-1)
        fallback = pprob.max(-1).values < self.predicate_presence_floor
        strength = strength.masked_fill(fallback[:, None, None], 0)
        coherent_map = (1 - strength[..., None]) * visual_map + strength[..., None] * predicate_map
        coherent_map = coherent_map / coherent_map.sum(-1, keepdim=True).clamp_min(1e-8)
        coherent_token = torch.einsum("bakl,bakn,blnd->bakd", layer_weights, coherent_map, projected_values)

        region_stack = torch.stack([ego_region_masks[name].to(field) for name in self.region_names])
        region_overlap = torch.einsum("bakn,rn->bakr", coherent_map, region_stack)
        region_id = region_overlap.argmax(-1)
        selected_region = region_stack[region_id]
        aggregate_value = torch.einsum("bakl,blnd->baknd", layer_weights, projected_values)
        background_weight = selected_region * (1 - coherent_map)
        background_weight = background_weight / background_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        background_token = torch.einsum("bakn,baknd->bakd", background_weight, aggregate_value)
        centered_token = self.center_norm(coherent_token - background_token)
        return {
            "visual_map": visual_map, "predicate_map": predicate_map, "coherent_map": coherent_map,
            "coherent_token": coherent_token, "centered_token": centered_token,
            "background_token": background_token, "background_region_id": region_id,
            "layer_weights": layer_weights, "projected_values": projected_values,
            "predicate_mixture": predicate_mixture,
            "predicate_top2_count": (predicate_mixture > 0).sum(-1),
            "predicate_agreement": agreement, "predicate_confidence": confidence,
            "predicate_strength": strength, "predicate_fallback": fallback,
        }
