from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


ACTION_EXPERT_TYPES = ["object", "lane", "drivable", "traffic_control", "global_context"]


def tokens_to_patch_map(tokens: torch.Tensor, image_height: int = 360, image_width: int = 640, patch_size: int = 8) -> torch.Tensor:
    h = image_height // patch_size
    w = image_width // patch_size
    expected = h * w
    if tokens.shape[1] == expected + 1:
        patch_tokens = tokens[:, 1:]
    elif tokens.shape[1] == expected:
        patch_tokens = tokens
    else:
        raise ValueError(f"Cannot map {tokens.shape[1]} tokens to patch grid {h}x{w}; expected {expected} or {expected + 1}.")
    return patch_tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], h, w).contiguous()


class ReferencePointSampler(nn.Module):
    def __init__(self, dim: int = 384, points_per_query: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.points_per_query = points_per_query
        base = torch.linspace(-0.04, 0.04, points_per_query)
        offsets = torch.stack([base, torch.flip(base, dims=[0])], dim=-1)
        self.offsets = nn.Parameter(offsets)

    def forward(self, patch_map: torch.Tensor, ref_points: torch.Tensor) -> torch.Tensor:
        b, d, _, _ = patch_map.shape
        if ref_points.dim() != 4:
            raise ValueError("ref_points must have shape [B, A, S, 2].")
        a, s = ref_points.shape[1], ref_points.shape[2]
        offsets = self.offsets[:s].view(1, 1, s, 2).to(ref_points.device, ref_points.dtype)
        grid = (ref_points + offsets).clamp(0.0, 1.0) * 2.0 - 1.0
        grid = grid.reshape(b, a * s, 1, 2)
        sampled = F.grid_sample(patch_map, grid, mode="bilinear", padding_mode="border", align_corners=False)
        sampled = sampled.squeeze(-1).transpose(1, 2).reshape(b, a, s, d)
        return sampled


class ActionEvidenceRouter(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, num_experts: int = 5, top_k: int = 2) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.net = nn.Sequential(nn.Linear(dim + 2, dim // 2), nn.GELU(), nn.Linear(dim // 2, num_experts))
        prior = torch.zeros(action_dim, num_experts)
        # forward: drivable + object; stop: object + traffic; left/right: lane + drivable.
        prior[0, 2] = 0.8
        prior[0, 0] = 0.6
        prior[1, 0] = 0.8
        prior[1, 3] = 0.8
        prior[2, 1] = 0.8
        prior[2, 2] = 0.6
        prior[3, 1] = 0.8
        prior[3, 2] = 0.6
        self.register_buffer("action_source_prior", prior)

    def forward(self, action_tokens: torch.Tensor, base_action_logits: torch.Tensor, action_uncertainty: torch.Tensor) -> dict[str, torch.Tensor]:
        prob = torch.sigmoid(base_action_logits).unsqueeze(-1)
        uncert = action_uncertainty.unsqueeze(-1)
        x = torch.cat([action_tokens, prob, uncert], dim=-1)
        logits = self.net(x) + self.action_source_prior.to(x.device, x.dtype).unsqueeze(0)
        top_idx = torch.topk(logits, k=min(self.top_k, self.num_experts), dim=-1).indices
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, top_idx, 1.0)
        masked_logits = logits.masked_fill(mask <= 0, -1e4)
        probs = torch.softmax(masked_logits, dim=-1) * mask
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-6)
        return {"expert_gate": torch.sigmoid(logits), "expert_route_mask": mask, "expert_route_probs": probs, "expert_logits": logits}


class _ActionSourceExpert(nn.Module):
    def __init__(self, dim: int = 384, source_id: int = 0, points_per_query: int = 4, geom_dim: int = 8) -> None:
        super().__init__()
        self.source_id = source_id
        self.points_per_query = points_per_query
        self.source_embed = nn.Parameter(torch.randn(dim) * 0.02)
        self.ref_head = nn.Sequential(nn.Linear(dim + 1, dim // 2), nn.GELU(), nn.Linear(dim // 2, points_per_query * 2))
        self.sampler = ReferencePointSampler(dim=dim, points_per_query=points_per_query)
        self.geom_proj = nn.Sequential(nn.Linear(geom_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.fuse = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.score = nn.Linear(dim, 1)

    def forward(
        self,
        action_tokens: torch.Tensor,
        patch_map: torch.Tensor,
        base_action_logits: torch.Tensor,
        bag_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        b, a, d = action_tokens.shape
        src = self.source_embed.to(action_tokens.device, action_tokens.dtype).view(1, 1, d)
        prob = torch.sigmoid(base_action_logits).unsqueeze(-1)
        ref = torch.sigmoid(self.ref_head(torch.cat([action_tokens + src, prob], dim=-1))).reshape(b, a, self.points_per_query, 2)
        sampled = self.sampler(patch_map, ref).mean(2)
        patch_tokens = patch_map.flatten(2).transpose(1, 2)
        patch_saliency = patch_tokens.norm(dim=-1)
        topk = min(self.points_per_query, patch_tokens.shape[1])
        top_idx = torch.topk(patch_saliency, k=topk, dim=1).indices
        salient = patch_tokens.gather(1, top_idx.unsqueeze(-1).expand(-1, -1, d)).mean(1)
        sampled = sampled + 0.25 * salient.unsqueeze(1)
        if bag_features is None:
            geom = torch.zeros(b, a, d, device=action_tokens.device, dtype=action_tokens.dtype)
            bag_count = torch.zeros(b, device=action_tokens.device, dtype=action_tokens.dtype)
        else:
            bag = bag_features.to(action_tokens.device, action_tokens.dtype)
            geom_feat = self.geom_proj(bag).mean(1)
            geom = geom_feat.unsqueeze(1).expand(-1, a, -1)
            bag_count = (bag.abs().sum(-1) > 0).sum(-1).float()
        evidence = torch.tanh(self.fuse(torch.cat([action_tokens + src, sampled, geom], dim=-1)))
        score = self.score(evidence).squeeze(-1)
        return {
            "evidence_tokens": evidence,
            "evidence_scores": score,
            "evidence_reliability": torch.sigmoid(score),
            "reference_points": ref,
            "bag_count": bag_count,
        }


class ActionObjectEvidenceExpert(_ActionSourceExpert):
    pass


class ActionLaneEvidenceExpert(_ActionSourceExpert):
    pass


class ActionDrivableEvidenceExpert(_ActionSourceExpert):
    pass


class ActionTrafficControlEvidenceExpert(_ActionSourceExpert):
    pass


class ActionGlobalContextEvidenceExpert(_ActionSourceExpert):
    def forward(self, action_tokens: torch.Tensor, patch_map: torch.Tensor, base_action_logits: torch.Tensor, bag_features: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        out = super().forward(action_tokens, patch_map, base_action_logits, bag_features)
        out["evidence_scores"] = out["evidence_scores"] * 0.5
        return out


class ActionEvidenceExpertBank(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, image_height: int = 360, image_width: int = 640, patch_size: int = 8) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.patch_size = patch_size
        self.router = ActionEvidenceRouter(dim=dim, action_dim=action_dim, top_k=2)
        self.experts = nn.ModuleDict({
            "object": ActionObjectEvidenceExpert(dim=dim, source_id=0),
            "lane": ActionLaneEvidenceExpert(dim=dim, source_id=1),
            "drivable": ActionDrivableEvidenceExpert(dim=dim, source_id=2),
            "traffic_control": ActionTrafficControlEvidenceExpert(dim=dim, source_id=3),
            "global_context": ActionGlobalContextEvidenceExpert(dim=dim, source_id=4),
        })
        self.delta_head = nn.Sequential(nn.Linear(dim + 2, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def forward(
        self,
        action_tokens: torch.Tensor,
        tokens: torch.Tensor,
        base_action_logits: torch.Tensor,
        action_uncertainty: torch.Tensor | None = None,
        structured_bags: dict[str, torch.Tensor] | None = None,
        structured: list[dict[str, Any] | None] | None = None,
    ) -> dict[str, Any]:
        action_uncertainty = action_uncertainty if action_uncertainty is not None else torch.zeros_like(base_action_logits)
        patch_map = tokens_to_patch_map(tokens, self.image_height, self.image_width, self.patch_size)
        route = self.router(action_tokens, base_action_logits, action_uncertainty)
        contexts = []
        scores = []
        reliabilities = []
        refs: dict[str, torch.Tensor] = {}
        bag_counts: dict[str, torch.Tensor] = {}
        for name in ACTION_EXPERT_TYPES:
            bag = structured_bags.get(name) if structured_bags is not None else None
            eout = self.experts[name](action_tokens, patch_map, base_action_logits, bag)
            contexts.append(eout["evidence_tokens"])
            scores.append(eout["evidence_scores"])
            reliabilities.append(eout["evidence_reliability"])
            refs[name] = eout["reference_points"]
            bag_counts[name] = eout["bag_count"]
        ctx_stack = torch.stack(contexts, dim=2)
        score_stack = torch.stack(scores, dim=2)
        rel_stack = torch.stack(reliabilities, dim=2)
        probs = route["expert_route_probs"].unsqueeze(-1)
        context = (ctx_stack * probs).sum(2)
        selected_scores = (score_stack * route["expert_route_probs"]).sum(2)
        random_scores = score_stack.mean(2)
        selected_rel = (rel_stack * route["expert_route_probs"]).sum(2)
        delta_raw = self.delta_head(torch.cat([context, torch.sigmoid(base_action_logits).unsqueeze(-1), action_uncertainty.unsqueeze(-1)], dim=-1)).squeeze(-1)
        usage = {name: int(route["expert_route_mask"][..., i].sum().detach().cpu().item()) for i, name in enumerate(ACTION_EXPERT_TYPES)}
        return {
            **route,
            "action_evidence_context": context,
            "action_evidence_delta_raw": delta_raw,
            "action_evidence_scores": selected_scores,
            "action_random_evidence_scores": random_scores,
            "action_evidence_reliability": selected_rel,
            "action_reference_points": refs,
            "action_expert_usage": usage,
            "action_expert_bag_count": bag_counts,
        }
