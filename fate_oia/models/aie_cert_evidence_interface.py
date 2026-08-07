from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .aie_cert_atom_transport import AIECertAtomTransport
from .aie_cert_deformable_reread import AIECertDeformableReread
from .aie_cert_predicate_bank import AIECertPredicateBank


class AIECertEvidenceInterface(nn.Module):
    REGION_NAMES = ("upper_traffic_region", "left_corridor", "right_corridor", "bottom_drivable_region", "front_center")

    def __init__(self, dim=384, action_dim=4, probes_per_action=4, num_layers=3, num_predicates=32,
                 grid_hw=(45, 80), points_per_layer=8, max_offset=0.25, transport_heads=4):
        super().__init__()
        self.dim, self.action_dim, self.atoms, self.num_layers = dim, action_dim, probes_per_action, num_layers
        self.grid_hw = grid_hw
        self.role = nn.Parameter(torch.randn(probes_per_action, dim) * 0.02)
        self.query_norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, dim)
        self.keys = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.values = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers))
        self.layer_router = nn.Linear(dim, num_layers)
        self.predicate_bank = AIECertPredicateBank(dim, num_predicates, 64)
        self.deformable = AIECertDeformableReread(dim, grid_hw, num_layers, points_per_layer, max_offset)
        self.transport = AIECertAtomTransport(dim, action_dim, probes_per_action, transport_heads)

    def _atom_regions(self, reference: Tensor, regions: dict[str, Tensor]) -> Tensor:
        coords = reference
        names = self.REGION_NAMES
        flat = torch.stack([regions[name] for name in names], 0).to(reference)
        h, w = self.grid_hw
        x = (coords[..., 0] * (w - 1)).round().long().clamp(0, w - 1)
        y = (coords[..., 1] * (h - 1)).round().long().clamp(0, h - 1)
        index = y * w + x
        score = flat[:, index].movedim(0, -1)
        chosen = score.argmax(-1)
        return flat[chosen]

    def forward(self, action_nodes: Tensor, field: Tensor, predicate_attention: Tensor,
                predicate_probs: Tensor, ego_regions: dict[str, Tensor], prior_scale=1.0, gamma_cap=0.25,
                local_reread_enabled=True, transport_enabled=True, background_center_enabled=True):
        field = field.detach()
        probes = self.query_norm(action_nodes.detach()[:, :, None] + self.role[None, None])
        layer_mix = torch.softmax(self.layer_router(probes), -1)
        visual_scores, values = [], []
        for layer in range(self.num_layers):
            key, value = self.keys[layer](field[:, layer]), self.values[layer](field[:, layer])
            visual_scores.append(torch.einsum("bakd,bnd->bakn", self.q(probes), key) / math.sqrt(self.dim))
            values.append(value)
        score_stack = torch.stack(visual_scores, 3)
        visual_score = (layer_mix[..., None] * score_stack).sum(3)
        visual_attention = torch.softmax(visual_score, -1)
        global_token = sum(torch.einsum("bakn,bnd->bakd", layer_mix[..., i, None] *
                           torch.softmax(visual_scores[i], -1), values[i]) for i in range(self.num_layers))
        bank = self.predicate_bank(global_token, predicate_probs.detach(), predicate_attention.detach())
        combined = visual_score + prior_scale * bank["predicate_prior_strength"][..., None] * bank["predicate_log_density_ratio"]
        map_pre = torch.softmax(combined, -1)
        reread = self.deformable(probes, field, map_pre, global_token)
        token_pre = global_token + (reread["local_token"] if local_reread_enabled else torch.zeros_like(global_token))
        transported = self.transport(token_pre, map_pre, gamma_cap=gamma_cap) if transport_enabled else {
            "atom_transport_matrix": torch.eye(self.atoms, device=field.device, dtype=field.dtype)[None,None].expand(field.shape[0],self.action_dim,-1,-1),
            "atom_transport_gamma": field.new_zeros(self.action_dim), "atom_token": token_pre, "atom_map": map_pre}
        region = self._atom_regions(reread["reference_point"], ego_regions)
        topk = min(64, transported["atom_map"].shape[-1])
        selected = torch.zeros_like(transported["atom_map"]).scatter_(-1,
            transported["atom_map"].topk(topk, -1).indices, 1.0)
        background_weight = region * (1.0 - selected)
        fallback = background_weight.sum(-1, keepdim=True) < 1e-6
        background_weight = torch.where(fallback, region, background_weight)
        background_weight = background_weight / background_weight.sum(-1, keepdim=True).clamp_min(1e-8)
        mixed_field = field.mean(1)
        background = torch.einsum("bakn,bnd->bakd", background_weight, mixed_field)
        centered = transported["atom_token"] - background if background_center_enabled else transported["atom_token"]
        return {"probe_queries": probes, "global_visual_score": visual_score,
            "global_attention": visual_attention, "global_token": global_token, "layer_mixture": layer_mix,
            **bank, "atom_map_pre_transport": map_pre, "atom_token_pre_transport": token_pre,
            **transported, **reread, "atom_region_mask": region, "background_token": background,
            "centered_atom_token": centered}
