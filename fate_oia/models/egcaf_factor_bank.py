from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from fate_oia.models.egcaf_dino_object_factors import DinoObjectLikeFactorGenerator
from fate_oia.models.egcaf_factor_types import FactorBatch, concatenate_factor_batches
from fate_oia.models.egcaf_scene_state_proxy import SceneStateProxyHead


class EgoCentricAnchorFactorGenerator(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_types: int = 11) -> None:
        super().__init__()
        self.names = ["front_center", "lower_center_drivable", "left_lane", "right_lane", "traffic_control", "global_context"]
        refs = torch.tensor([[0.50,0.55],[0.50,0.78],[0.32,0.68],[0.68,0.68],[0.50,0.25],[0.50,0.50]])
        self.register_buffer("reference_points", refs)
        self.anchor_queries = nn.Parameter(torch.randn(6, hidden_dim) * 0.02)
        self.offset_mlp = nn.Linear(hidden_dim, 2)
        self.type_head = nn.Linear(hidden_dim, num_types)

    def forward(self, p1: torch.Tensor) -> FactorBatch:
        b, c, h, w = p1.shape
        centers = (self.reference_points + torch.tanh(self.offset_mlp(self.anchor_queries)) * 0.08).clamp(0, 1)
        grid_y = torch.linspace(0, 1, h, device=p1.device).view(1,1,h,1)
        grid_x = torch.linspace(0, 1, w, device=p1.device).view(1,1,1,w)
        masks = torch.exp(-((grid_x-centers[:,0].view(1,-1,1,1))**2 + (grid_y-centers[:,1].view(1,-1,1,1))**2)/0.018).expand(b,-1,-1,-1)
        emb = torch.einsum("bmhw,bchw->bmc", masks/(masks.sum((-1,-2), keepdim=True)+1e-6), p1) + self.anchor_queries.unsqueeze(0)
        boxes = torch.stack([(centers[:,0]-0.14).clamp(0,1),(centers[:,1]-0.14).clamp(0,1),(centers[:,0]+0.14).clamp(0,1),(centers[:,1]+0.14).clamp(0,1)], -1).unsqueeze(0).expand(b,-1,-1)
        src = torch.zeros(b, 6, dtype=torch.long, device=p1.device)
        typ = self.type_head(emb)
        for i, type_id in enumerate([1,2,3,4,5,8]):
            typ[:, i, type_id] += 2.0
        rel = torch.full((b, 6), 0.75, device=p1.device)
        valid = torch.ones(b, 6, dtype=torch.bool, device=p1.device)
        return FactorBatch(emb, masks, boxes, src, typ, rel, valid, {"anchor_names": self.names, "learnable_reference": True})


class DrivingFactorCandidateBank(nn.Module):
    def __init__(self, hidden_dim: int = 256, object_factors: int = 8, num_types: int = 11) -> None:
        super().__init__()
        self.anchors = EgoCentricAnchorFactorGenerator(hidden_dim, num_types)
        self.objects = DinoObjectLikeFactorGenerator(hidden_dim, object_factors, num_types)
        self.scene = SceneStateProxyHead(hidden_dim, num_types)
        self.global_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.global_type = nn.Linear(hidden_dim, num_types)

    def forward(self, pyramid: list[dict[str, torch.Tensor]]) -> dict[str, object]:
        p1 = torch.stack([p["P1"] for p in pyramid], dim=1).mean(1)
        b, c, h, w = p1.shape
        anchor = self.anchors(p1)
        obj = self.objects(p1)
        scene_out = self.scene(p1)
        scene_factors = scene_out["scene_state_factors"]
        gemb = F.adaptive_avg_pool2d(p1, 1).flatten(1).unsqueeze(1) + self.global_query.view(1,1,-1)
        gmask = torch.ones(b, 1, h, w, device=p1.device)
        gbox = torch.tensor([0,0,1,1], dtype=p1.dtype, device=p1.device).view(1,1,4).expand(b,1,4)
        gsrc = torch.full((b,1), 3, dtype=torch.long, device=p1.device)
        gtyp = self.global_type(gemb); gtyp[...,8] += 2.0
        glob = FactorBatch(gemb, gmask, gbox, gsrc, gtyp, torch.ones(b,1,device=p1.device)*0.8, torch.ones(b,1,dtype=torch.bool,device=p1.device), {"source": "global"})
        return {
            "factors": concatenate_factor_batches([anchor, obj, scene_factors, glob]),
            "scene_state_logits": scene_out["scene_state_logits"],
            "scene_state_tokens": scene_out["scene_state_tokens"],
        }
