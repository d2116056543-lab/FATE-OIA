from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.egcaf_factor_types import FactorBatch


class SceneStateProxyHead(nn.Module):
    """Weak BDD100K scene-state proxy head.

    The model predicts six weak scene states. Training may attach weak labels
    from BDD100K object/lane/drivable geometry through train_egcaf_oia.py.
    Test forward never consumes GT scene labels.
    """

    def __init__(self, hidden_dim: int = 256, num_types: int = 11, num_states: int = 6) -> None:
        super().__init__()
        self.num_states = int(num_states)
        self.state_names = ["traffic_control", "front_object", "vulnerable_user", "left_lane", "right_lane", "drivable"]
        self.head = nn.Linear(hidden_dim, num_states)
        self.state_queries = nn.Parameter(torch.randn(num_states, hidden_dim) * 0.02)
        self.type_head = nn.Linear(hidden_dim, num_types)

    def forward(self, p1: torch.Tensor, weak_scene_state: torch.Tensor | None = None) -> dict[str, torch.Tensor | FactorBatch]:
        b, c, h, w = p1.shape
        pooled = p1.mean((-1, -2))
        logits = self.head(pooled)
        grid_y = torch.linspace(0, 1, h, device=p1.device).view(1, 1, h, 1)
        grid_x = torch.linspace(0, 1, w, device=p1.device).view(1, 1, 1, w)
        centers = torch.tensor(
            [[0.50, 0.25], [0.50, 0.50], [0.50, 0.55], [0.30, 0.67], [0.70, 0.67], [0.50, 0.80]],
            device=p1.device,
            dtype=p1.dtype,
        )
        masks = torch.exp(-((grid_x - centers[:, 0].view(1, -1, 1, 1)) ** 2 + (grid_y - centers[:, 1].view(1, -1, 1, 1)) ** 2) / 0.035)
        masks = masks.expand(b, -1, -1, -1)
        emb = torch.einsum("bshw,bchw->bsc", masks / (masks.sum((-1, -2), keepdim=True) + 1e-6), p1) + self.state_queries.unsqueeze(0)
        boxes = torch.stack(
            [
                (centers[:, 0] - 0.16).clamp(0, 1),
                (centers[:, 1] - 0.16).clamp(0, 1),
                (centers[:, 0] + 0.16).clamp(0, 1),
                (centers[:, 1] + 0.16).clamp(0, 1),
            ],
            -1,
        ).unsqueeze(0).expand(b, -1, -1)
        src = torch.full((b, self.num_states), 2, dtype=torch.long, device=p1.device)
        typ = self.type_head(emb)
        for i, type_id in enumerate([5, 6, 7, 3, 4, 2]):
            typ[:, i, type_id] += 2.0
        rel = torch.sigmoid(logits.detach())
        if weak_scene_state is not None:
            rel = torch.maximum(rel, weak_scene_state.float().clamp(0, 1))
        valid = torch.ones(b, self.num_states, dtype=torch.bool, device=p1.device)
        factors = FactorBatch(
            emb,
            masks,
            boxes,
            src,
            typ,
            rel,
            valid,
            {"source": "bdd100k_scene_state_proxy", "weak_labels_connected": weak_scene_state is not None},
            torch.full((b, self.num_states), -1, dtype=torch.long, device=p1.device),
        )
        return {
            "scene_state_logits": logits,
            "scene_state_tokens": emb,
            "scene_state_factors": factors,
            "scene_state_weak_labels_used": weak_scene_state is not None,
        }

