from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.sure_relation_attention import SURESparseRelationAttention
from fate_oia.models.sure_relation_memory import UncertaintyRelationMemory
from fate_oia.models.sure_relation_proposer import SURERelationProposer
from fate_oia.models.sure_residual_refiner import SUREBoundedResidualRefiner


class SUREOIAFeatureModel(nn.Module):
    """SURE-OIA v2 direct-image head.

    Primary logits are computed from image patch tokens only. Structured BDD100K
    evidence is exposed through gt_scene_upper logits for diagnostics/upper bound.
    """

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        relation_queries: int = 32,
        max_edges_per_label: int = 8,
        max_edges_total: int = 96,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.label_count = action_dim + reason_dim
        self.base = FATEOIAFeatureModel(dim=dim, action_dim=action_dim, reason_dim=reason_dim, use_label_query=True)
        self.proposer = SURERelationProposer(dim=dim, relation_queries=relation_queries)
        self.relation_attention = SURESparseRelationAttention(dim=dim, label_count=self.label_count, max_edges_per_label=max_edges_per_label, max_edges_total=max_edges_total)
        self.memory = UncertaintyRelationMemory(dim=dim)
        self.refiner = SUREBoundedResidualRefiner(dim=dim, action_dim=action_dim, reason_dim=reason_dim)
        self.context_norm = nn.LayerNorm(dim)
        self.gt_upper_head = nn.Linear(dim, self.label_count)

    def forward(
        self,
        tokens: torch.Tensor,
        structured: list[dict[str, Any]] | None = None,
        image_meta: list[dict[str, Any]] | None = None,
        return_gt_scene_upper: bool = True,
    ) -> dict[str, Any]:
        base_out = self.base(tokens)
        label_tokens = base_out["label_tokens"]
        action_base = base_out["action_fused_logits"]
        reason_base = base_out["reason_logits"]
        base_logits = torch.cat([action_base, reason_base], dim=1)

        rel = self.proposer(tokens, structured=structured, image_meta=image_meta, use_gt_scene_upper=False)
        sparse = self.relation_attention(label_tokens, rel["relation_tokens"], rel["relation_type_ids"])
        mem = self.memory(label_tokens + sparse["label_context"], base_logits)
        updated_label_tokens = label_tokens + sparse["label_context"] + mem["memory_context"]
        refined = self.refiner(updated_label_tokens, action_base, reason_base)

        action_final = refined["action_logits"]
        reason_final = refined["reason_logits"]
        logits_final = torch.cat([action_final, reason_final], dim=1)
        out: dict[str, Any] = {
            **base_out,
            "action_logits": action_final,
            "action_final_logits": action_final,
            "reason_logits": reason_final,
            "reason_final_logits": reason_final,
            "action_base_logits": action_base,
            "reason_base_logits": reason_base,
            "logits": logits_final,
            "label_tokens_sure": updated_label_tokens,
            "relation_tokens": rel["relation_tokens"],
            "relation_scores": rel["relation_scores"],
            "selected_relation_indices": sparse["selected_relation_indices"],
            "memory_gate": mem["memory_gate"],
            "relation_stats": {**rel["stats"], **sparse["stats"]},
            "gradnorm_stats": {},
            "action_safe_stats": refined["action_safe_stats"],
        }
        if return_gt_scene_upper:
            gt_rel = self.proposer(tokens, structured=structured, image_meta=image_meta, use_gt_scene_upper=True)
            gt_context = self.context_norm(gt_rel["relation_tokens"].mean(1))
            gt_logits = self.gt_upper_head(gt_context)
            out["action_gt_scene_upper_logits"] = gt_logits[:, : self.action_dim]
            out["reason_gt_scene_upper_logits"] = gt_logits[:, self.action_dim :]
            out["gt_scene_upper_stats"] = gt_rel["stats"]
        return out
