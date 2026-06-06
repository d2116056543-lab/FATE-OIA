from __future__ import annotations

import torch
from torch import nn

from fate_oia.models.diva_action_set_transformer import ActionSetRelationTransformer
from fate_oia.models.diva_deformable_attention import MultiScaleDeformableSampler


EVIDENCE_NAMES = ["front_center", "lower_center_drivable", "left_lane", "right_lane", "traffic_control", "global_context"]
REFERENCE_POINTS = torch.tensor([
    [0.50, 0.55],
    [0.50, 0.78],
    [0.32, 0.68],
    [0.68, 0.68],
    [0.50, 0.25],
    [0.50, 0.50],
], dtype=torch.float32)
ACTION_TO_EVIDENCE = torch.tensor([
    [1, 0, 5],  # forward
    [0, 4, 5],  # stop
    [2, 0, 5],  # left
    [3, 0, 5],  # right
], dtype=torch.long)


class EgoEvidenceLatentActor(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, num_points: int = 4) -> None:
        super().__init__()
        self.dim = int(dim)
        self.action_dim = int(action_dim)
        self.evidence_queries = nn.Parameter(torch.randn(len(EVIDENCE_NAMES), dim) * 0.02)
        self.sampler = MultiScaleDeformableSampler(dim, num_scales=3, num_points=num_points)
        self.confidence = nn.Linear(dim, 1)
        self.register_buffer("reference_points", REFERENCE_POINTS)
        self.register_buffer("action_to_evidence", ACTION_TO_EVIDENCE)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        b = features["P1"].shape[0]
        all_action_tokens = []
        all_conf = []
        sample_points = []
        named_contexts = []
        query = self.evidence_queries.unsqueeze(0).expand(b, -1, -1)
        refs = self.reference_points.unsqueeze(0).expand(b, -1, -1)
        for a in range(self.action_dim):
            feats = [features["P1"][:, a], features["P2"][:, a], features["P3"][:, a]]
            sampled = self.sampler(query, feats, refs)
            contexts = sampled["context"]
            named_contexts.append(contexts)
            idx = self.action_to_evidence[a]
            action_tokens = contexts[:, idx]
            all_action_tokens.append(action_tokens)
            all_conf.append(torch.sigmoid(self.confidence(action_tokens.mean(1))).squeeze(-1))
            sample_points.append(sampled["sample_points"][:, idx])
        action_evidence = torch.stack(all_action_tokens, dim=1)
        evidence_conf = torch.stack(all_conf, dim=1)
        return {
            "action_evidence_tokens": action_evidence,
            "evidence_tokens_named": torch.stack(named_contexts, dim=1).mean(1),
            "evidence_confidence": evidence_conf,
            "evidence_sample_points": torch.stack(sample_points, dim=1),
            "evidence_names": EVIDENCE_NAMES,
        }


class DIVAVisualActor(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, num_heads: int = 4) -> None:
        super().__init__()
        self.ego_actor = EgoEvidenceLatentActor(dim, action_dim)
        self.action_set = ActionSetRelationTransformer(dim=dim, action_dim=action_dim, depth=2, num_heads=num_heads)

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        evidence = self.ego_actor(features)
        action_tokens_in = evidence["action_evidence_tokens"].mean(dim=2)
        rel = self.action_set(action_tokens_in)
        return {**evidence, **rel, "z_eva": rel["z_eva"]}
