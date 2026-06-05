from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from fate_oia.models.egcaf_dino_object_factors import DinoObjectLikeFactorGenerator
from fate_oia.models.egcaf_factor_types import FactorBatch, concatenate_factor_batches
from fate_oia.models.egcaf_scene_state_proxy import SceneStateProxyHead


class EgoCentricAnchorFactorGenerator(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_types: int = 11, num_actions: int = 4, points_per_scale: int = 4) -> None:
        super().__init__()
        self.names = ["front_center", "lower_center_drivable", "left_lane", "right_lane", "traffic_control", "global_context"]
        refs = torch.tensor([[0.50, 0.55], [0.50, 0.78], [0.32, 0.68], [0.68, 0.68], [0.50, 0.25], [0.50, 0.50]])
        self.register_buffer("reference_points", refs)
        self.num_actions = int(num_actions)
        self.points_per_scale = int(points_per_scale)
        self.anchor_queries = nn.Parameter(torch.randn(len(self.names), hidden_dim) * 0.02)
        self.learned_offsets = nn.Parameter(torch.zeros(len(self.names), 3, points_per_scale, 2))
        self.scale_logits = nn.Parameter(torch.zeros(len(self.names), 3))
        self.type_head = nn.Linear(hidden_dim, num_types)

    @staticmethod
    def _sample_points(feature: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        # feature [B,C,H,W], points [N,P,2] in [0,1] xy.
        b, c, _, _ = feature.shape
        n, p, _ = points.shape
        grid = points.clone()
        grid = grid * 2 - 1
        grid = grid.view(1, n * p, 1, 2).expand(b, -1, -1, -1)
        sampled = F.grid_sample(feature, grid, align_corners=True, mode="bilinear").squeeze(-1).transpose(1, 2)
        return sampled.view(b, n, p, c)

    def forward(self, pyramid_by_action: list[dict[str, torch.Tensor]]) -> FactorBatch:
        factors: list[FactorBatch] = []
        for action_id, pyramid in enumerate(pyramid_by_action):
            p1, p2, p3 = pyramid["P1"], pyramid["P2"], pyramid["P3"]
            b, c, h, w = p1.shape
            base = self.reference_points.to(p1.device, p1.dtype)
            offsets = torch.tanh(self.learned_offsets.to(p1.device, p1.dtype)) * 0.10
            points = (base[:, None, None, :] + offsets).clamp(0, 1)
            scale_samples = []
            for scale_i, feat in enumerate([p1, p2, p3]):
                scale_samples.append(self._sample_points(feat, points[:, scale_i]))
            scale_w = torch.softmax(self.scale_logits.to(p1.device, p1.dtype), dim=-1)
            emb = sum(scale_samples[i].mean(2) * scale_w[:, i].view(1, -1, 1) for i in range(3))
            emb = emb + self.anchor_queries.to(p1.device, p1.dtype).unsqueeze(0)
            grid_y = torch.linspace(0, 1, h, device=p1.device).view(1, 1, h, 1)
            grid_x = torch.linspace(0, 1, w, device=p1.device).view(1, 1, 1, w)
            centers = points.mean(2).mean(1)
            masks = torch.exp(-((grid_x - centers[:, 0].view(1, -1, 1, 1)) ** 2 + (grid_y - centers[:, 1].view(1, -1, 1, 1)) ** 2) / 0.020)
            masks = masks.expand(b, -1, -1, -1)
            boxes = torch.stack(
                [
                    (centers[:, 0] - 0.14).clamp(0, 1),
                    (centers[:, 1] - 0.14).clamp(0, 1),
                    (centers[:, 0] + 0.14).clamp(0, 1),
                    (centers[:, 1] + 0.14).clamp(0, 1),
                ],
                -1,
            ).unsqueeze(0).expand(b, -1, -1)
            src = torch.zeros(b, len(self.names), dtype=torch.long, device=p1.device)
            typ = self.type_head(emb)
            for i, type_id in enumerate([1, 2, 3, 4, 5, 8]):
                typ[:, i, type_id] += 2.0
            rel = torch.full((b, len(self.names)), 0.75, device=p1.device)
            valid = torch.ones(b, len(self.names), dtype=torch.bool, device=p1.device)
            action_ids = torch.full((b, len(self.names)), action_id, dtype=torch.long, device=p1.device)
            factors.append(
                FactorBatch(
                    emb,
                    masks,
                    boxes,
                    src,
                    typ,
                    rel,
                    valid,
                    {
                        "anchor_names": self.names,
                        "action_conditioned": True,
                        "multi_scale_sampling": True,
                        "points_per_scale": self.points_per_scale,
                    },
                    action_ids,
                )
            )
        return concatenate_factor_batches(factors)


class DrivingFactorCandidateBank(nn.Module):
    def __init__(self, hidden_dim: int = 256, object_factors: int = 8, num_types: int = 11, num_actions: int = 4) -> None:
        super().__init__()
        self.anchors = EgoCentricAnchorFactorGenerator(hidden_dim, num_types, num_actions=num_actions)
        self.objects = DinoObjectLikeFactorGenerator(hidden_dim, object_factors, num_types)
        self.scene = SceneStateProxyHead(hidden_dim, num_types)
        self.global_query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.global_type = nn.Linear(hidden_dim, num_types)

    def forward(self, pyramid: list[dict[str, torch.Tensor]], weak_scene_state: torch.Tensor | None = None) -> dict[str, object]:
        shared_p1 = torch.stack([p["P1"] for p in pyramid], dim=1).mean(1)
        b, c, h, w = shared_p1.shape
        anchor = self.anchors(pyramid)
        obj = self.objects(shared_p1)
        scene_out = self.scene(shared_p1, weak_scene_state=weak_scene_state)
        scene_factors = scene_out["scene_state_factors"]
        gemb = F.adaptive_avg_pool2d(shared_p1, 1).flatten(1).unsqueeze(1) + self.global_query.view(1, 1, -1)
        gmask = torch.ones(b, 1, h, w, device=shared_p1.device)
        gbox = torch.tensor([0, 0, 1, 1], dtype=shared_p1.dtype, device=shared_p1.device).view(1, 1, 4).expand(b, 1, 4)
        gsrc = torch.full((b, 1), 3, dtype=torch.long, device=shared_p1.device)
        gtyp = self.global_type(gemb)
        gtyp[..., 8] += 2.0
        glob = FactorBatch(
            gemb,
            gmask,
            gbox,
            gsrc,
            gtyp,
            torch.ones(b, 1, device=shared_p1.device) * 0.8,
            torch.ones(b, 1, dtype=torch.bool, device=shared_p1.device),
            {"source": "global"},
            torch.full((b, 1), -1, dtype=torch.long, device=shared_p1.device),
        )
        return {
            "factors": concatenate_factor_batches([anchor, obj, scene_factors, glob]),
            "shared_factors": concatenate_factor_batches([obj, scene_factors, glob]),
            "action_conditioned_factors": anchor,
            "scene_state_logits": scene_out["scene_state_logits"],
            "scene_state_tokens": scene_out["scene_state_tokens"],
            "scene_state_weak_labels_used": scene_out["scene_state_weak_labels_used"],
            "factor_bank_stats": {
                "action_conditioned_factor_count": int(anchor.embeddings.shape[1]),
                "shared_factor_count": int(obj.embeddings.shape[1] + scene_factors.embeddings.shape[1] + glob.embeddings.shape[1]),
                "action_specific_maps_preserved": True,
                "no_action_map_mean_before_anchor": True,
            },
        }

