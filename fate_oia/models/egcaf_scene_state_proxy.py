from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from fate_oia.models.egcaf_factor_types import FactorBatch


class SceneStateProxyHead(nn.Module):
    STATE_NAMES = [
        "traffic_control_present", "front_vehicle_or_obstacle", "vulnerable_user_front",
        "left_lane_structure", "right_lane_structure", "lower_center_drivable",
    ]

    def __init__(self, hidden_dim: int = 256, num_types: int = 11) -> None:
        super().__init__()
        self.state_head = nn.Linear(hidden_dim, len(self.STATE_NAMES))
        self.token_proj = nn.Linear(hidden_dim, hidden_dim)
        self.type_head = nn.Linear(hidden_dim, num_types)

    def forward(self, p1: torch.Tensor) -> dict[str, torch.Tensor | FactorBatch]:
        b, c, h, w = p1.shape
        pooled = F.adaptive_avg_pool2d(p1, 1).flatten(1)
        logits = self.state_head(pooled)
        centers = torch.tensor([[0.5,0.25],[0.5,0.50],[0.5,0.65],[0.32,0.68],[0.68,0.68],[0.5,0.78]], device=p1.device)
        grid_y = torch.linspace(0, 1, h, device=p1.device).view(1, 1, h, 1)
        grid_x = torch.linspace(0, 1, w, device=p1.device).view(1, 1, 1, w)
        masks = torch.exp(-((grid_x - centers[:,0].view(1,-1,1,1))**2 + (grid_y - centers[:,1].view(1,-1,1,1))**2) / 0.025).expand(b,-1,-1,-1)
        emb = torch.einsum("bmhw,bchw->bmc", masks / (masks.sum((-1,-2), keepdim=True)+1e-6), p1)
        emb = self.token_proj(emb)
        boxes = torch.stack([(centers[:,0]-0.12).clamp(0,1),(centers[:,1]-0.12).clamp(0,1),(centers[:,0]+0.12).clamp(0,1),(centers[:,1]+0.12).clamp(0,1)], -1).unsqueeze(0).expand(b,-1,-1)
        src = torch.full((b, 6), 2, dtype=torch.long, device=p1.device)
        typ = self.type_head(emb)
        rel = torch.sigmoid(logits)
        valid = torch.ones(b, 6, dtype=torch.bool, device=p1.device)
        factors = FactorBatch(emb, masks, boxes, src, typ, rel, valid, {"weak_proxy": True, "uses_gt_in_test_forward": False})
        return {"scene_state_logits": logits, "scene_state_tokens": emb, "scene_state_factors": factors}
