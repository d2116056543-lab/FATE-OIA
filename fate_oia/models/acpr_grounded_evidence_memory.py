from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from .acpr_sparse_ops import entmax15_bisect


@dataclass(frozen=True)
class ACPREvidenceSlotSpec:
    id: int
    name: str
    group: str
    grounding_sources: tuple[str, ...]
    region_prior: str
    use_object_boxes: bool
    use_lane_polylines: bool
    use_drivable_masks: bool
    oracle_allowed: bool
    reliability: float


def load_evidence_slot_specs(path: str | Path) -> list[ACPREvidenceSlotSpec]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    slots = list(data.get("slots", []))
    specs: list[ACPREvidenceSlotSpec] = []
    for i, row in enumerate(slots):
        specs.append(
            ACPREvidenceSlotSpec(
                id=int(row.get("id", i)),
                name=str(row["name"]),
                group=str(row["group"]),
                grounding_sources=tuple(str(x) for x in row.get("grounding_sources", [])),
                region_prior=str(row.get("region_prior", "global")),
                use_object_boxes=bool(row.get("use_object_boxes", False)),
                use_lane_polylines=bool(row.get("use_lane_polylines", False)),
                use_drivable_masks=bool(row.get("use_drivable_masks", False)),
                oracle_allowed=bool(row.get("oracle_allowed", False)),
                reliability=float(row.get("reliability", 0.5)),
            )
        )
    if not 16 <= len(specs) <= 24:
        raise ValueError(f"ACPR-GEM evidence slots must be 16-24, got {len(specs)}")
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("ACPR-GEM evidence slot names must be unique")
    return specs


def _region_prior(name: str, grid_hw: tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    h, w = grid_hw
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, h, device=device, dtype=dtype),
        torch.linspace(0, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    front_center = torch.exp(-(((xx - 0.5) ** 2) / 0.08 + ((yy - 0.75) ** 2) / 0.20))
    left_corridor = torch.sigmoid((0.45 - xx) * 10.0) * torch.sigmoid((yy - 0.35) * 10.0)
    right_corridor = torch.sigmoid((xx - 0.55) * 10.0) * torch.sigmoid((yy - 0.35) * 10.0)
    upper_traffic_region = torch.sigmoid((0.45 - yy) * 10.0)
    bottom_drivable_region = yy
    priors = {
        "front_center": front_center,
        "left_corridor": left_corridor,
        "right_corridor": right_corridor,
        "upper_traffic_region": upper_traffic_region,
        "bottom_drivable_region": bottom_drivable_region,
        "global": torch.ones_like(xx),
    }
    prior = priors.get(name, priors["global"]).reshape(-1)
    return prior / prior.max().clamp_min(1e-6)


class ACPRGroundedEvidencePooler(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        slot_specs: list[ACPREvidenceSlotSpec] | None = None,
        slots_config: str | Path = "configs/acpr_gem_evidence_slots.yaml",
        grid_hw: tuple[int, int] = (45, 80),
        topk: int = 256,
        prior_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.slot_specs = list(slot_specs or load_evidence_slot_specs(slots_config))
        self.num_slots = len(self.slot_specs)
        self.dim = int(dim)
        self.grid_hw = grid_hw
        self.topk = int(topk)
        self.prior_weight = float(prior_weight)
        self.evidence_queries = nn.Parameter(torch.randn(self.num_slots, dim) * 0.02)
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

    @property
    def slot_names(self) -> list[str]:
        return [s.name for s in self.slot_specs]

    @property
    def slot_groups(self) -> list[str]:
        return [s.group for s in self.slot_specs]

    def spatial_prior(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([_region_prior(s.region_prior, self.grid_hw, device, dtype) for s in self.slot_specs], dim=0)

    def _sparse_attention(self, score: torch.Tensor) -> torch.Tensor:
        if self.topk > 0 and self.topk < score.shape[-1]:
            vals, idx = torch.topk(score, k=self.topk, dim=-1)
            attn_small = entmax15_bisect(vals, dim=-1)
            attn = torch.zeros_like(score)
            return attn.scatter(-1, idx, attn_small)
        return entmax15_bisect(score, dim=-1)

    def forward(
        self,
        patch_tokens_by_layer: torch.Tensor,
        grounding_targets: torch.Tensor | None = None,
        grounding_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        patch_tokens = patch_tokens_by_layer.mean(1) if patch_tokens_by_layer.dim() == 4 else patch_tokens_by_layer
        b, n, d = patch_tokens.shape
        q = self.query_proj(self.evidence_queries).view(1, self.num_slots, d)
        k = self.key_proj(patch_tokens)
        v = self.value_proj(patch_tokens)
        score = torch.einsum("bmd,bnd->bmn", q.expand(b, -1, -1), k) / (d**0.5)
        prior = self.spatial_prior(score.device, score.dtype)
        if prior.shape[-1] == n:
            score = score + self.prior_weight * prior.clamp_min(1e-6).log().unsqueeze(0)
        attn = self._sparse_attention(score)
        tokens = torch.einsum("bmn,bnd->bmd", attn, v)
        support = (attn > 1e-5).float().sum(-1)
        entropy = -(attn.clamp_min(1e-9).log() * attn).sum(-1)
        if grounding_targets is None:
            grounding_targets = torch.zeros(b, self.num_slots, n, device=score.device, dtype=score.dtype)
        if grounding_mask is None:
            grounding_mask = torch.zeros(b, self.num_slots, device=score.device, dtype=score.dtype)
        mass = (attn * grounding_targets.to(attn.device, attn.dtype)).sum(-1)
        stats = {
            "evidence_support_size_mean": float(support.mean().detach().cpu()),
            "evidence_entropy_mean": float(entropy.mean().detach().cpu()),
            "evidence_grounded_mass_mean": float((mass * grounding_mask).sum().detach().cpu() / grounding_mask.sum().clamp_min(1).detach().cpu()),
        }
        return {
            "evidence_tokens": tokens,
            "evidence_attention": attn,
            "evidence_scores": score,
            "evidence_slot_names": self.slot_names,
            "evidence_slot_groups": self.slot_groups,
            "evidence_grounding_targets": grounding_targets,
            "evidence_grounding_mask": grounding_mask,
            "evidence_available_rate": grounding_mask.float().mean(),
            "evidence_stats": stats,
            "evidence_oracle_mode": False,
        }


class ACPREvidenceOraclePooler(nn.Module):
    def __init__(self, learned_pooler: ACPRGroundedEvidencePooler) -> None:
        super().__init__()
        self.learned_pooler = learned_pooler

    def forward(self, patch_tokens: torch.Tensor, masks: torch.Tensor) -> dict[str, Any]:
        if patch_tokens.dim() == 4:
            patch_tokens = patch_tokens.mean(1)
        learned = self.learned_pooler(patch_tokens)
        m = masks.to(patch_tokens.device, patch_tokens.dtype)
        available = m.sum(-1) > 0
        norm = m / m.sum(-1, keepdim=True).clamp_min(1e-6)
        oracle_tokens = torch.einsum("bmn,bnd->bmd", norm, patch_tokens)
        tokens = torch.where(available.unsqueeze(-1), oracle_tokens, learned["evidence_tokens"])
        attn = torch.where(available.unsqueeze(-1), norm, learned["evidence_attention"])
        learned.update(
            {
                "evidence_tokens": tokens,
                "evidence_attention": attn,
                "oracle_available": available,
                "evidence_oracle_mode": True,
            }
        )
        return learned


class ACPREvidenceMemoryAugmenter(nn.Module):
    def __init__(self, dim: int = 384, max_delta: float = 0.20, num_heads: int = 4) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
        self.evidence_out_proj = nn.Linear(dim, dim)
        self.max_delta = float(max_delta)
        nn.init.zeros_(self.evidence_out_proj.weight)
        nn.init.zeros_(self.evidence_out_proj.bias)

    def forward(self, nodes: torch.Tensor, evidence_tokens: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if evidence_tokens is None:
            b, l, d = nodes.shape
            return nodes, nodes.new_zeros(b, l, d), nodes.new_zeros(b, l, 0)
        context, attn = self.cross_attn(nodes, evidence_tokens, evidence_tokens, need_weights=True)
        raw = self.evidence_out_proj(context)
        delta = self.max_delta * torch.tanh(raw / max(self.max_delta, 1e-6))
        return nodes + delta, context, attn


class ACPREvidenceGroundingLoss(nn.Module):
    def __init__(self, entropy_weight: float = 0.02) -> None:
        super().__init__()
        self.entropy_weight = float(entropy_weight)

    def forward(
        self,
        attention: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if targets.numel() == 0 or mask.sum() <= 0:
            return attention.sum() * 0.0
        targets = targets.to(attention.device, attention.dtype)
        mask = mask.to(attention.device, attention.dtype)
        mass = (attention * targets).sum(-1).clamp_min(1e-8)
        entropy = -(attention.clamp_min(1e-9).log() * attention).sum(-1)
        loss = -mass.log() + self.entropy_weight * entropy
        if scores is not None:
            score_targets = targets > 0
            target_scores = scores.masked_fill(~score_targets, -1.0e4)
            # This score-level objective keeps gradients available even when
            # sparse top-k attention has not yet selected the grounded region.
            score_loss = torch.logsumexp(scores, dim=-1) - torch.logsumexp(target_scores, dim=-1)
            loss = 0.5 * loss + score_loss
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
