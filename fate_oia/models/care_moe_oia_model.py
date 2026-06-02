from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.care_action_reason_router import ActionReasonRouter
from fate_oia.models.care_expert_router import EXPERT_TYPES, ExpertRouter
from fate_oia.models.care_evidence_experts import (
    DrivableEvidenceExpert,
    GlobalContextEvidenceExpert,
    LaneEvidenceExpert,
    ObjectEvidenceExpert,
    TrafficControlEvidenceExpert,
)
from fate_oia.models.care_evidence_bag import EvidenceBagBuilder
from fate_oia.models.care_reason_update import EvidenceToReasonUpdate
from fate_oia.models.care_action_feedback import ReasonToActionSafeFeedback


class CAREMoEOIAModel(nn.Module):
    def __init__(
        self,
        base_fate: FATEOIAFeatureModel | None = None,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        test_top_k_reasons: int = 12,
        action_cap_max: float = 0.04,
        action_feedback_warmup_epochs: int = 2,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.base_fate = base_fate or FATEOIAFeatureModel(dim=dim, action_dim=action_dim, reason_dim=reason_dim, use_label_query=True)
        self.reason_router = ActionReasonRouter(action_dim=action_dim, reason_dim=reason_dim, dim=dim, test_top_k=test_top_k_reasons)
        self.expert_router = ExpertRouter(dim=dim, action_dim=action_dim, reason_dim=reason_dim, top_k=2)
        self.experts = nn.ModuleDict({
            "object": ObjectEvidenceExpert(dim=dim),
            "lane": LaneEvidenceExpert(dim=dim),
            "drivable": DrivableEvidenceExpert(dim=dim),
            "traffic_control": TrafficControlEvidenceExpert(dim=dim),
            "global_context": GlobalContextEvidenceExpert(dim=dim),
        })
        self.bag_builder = EvidenceBagBuilder()
        self.reason_update = EvidenceToReasonUpdate(dim=dim, reason_dim=reason_dim)
        self.action_feedback = ReasonToActionSafeFeedback(action_dim=action_dim, reason_dim=reason_dim, cap=action_cap_max, warmup_epochs=action_feedback_warmup_epochs)

    def forward(
        self,
        tokens: torch.Tensor,
        batch: dict[str, Any] | None = None,
        structured: list[dict[str, Any] | None] | None = None,
        epoch: int = 0,
        force_structured_upper: bool = False,
        return_diagnostics: bool = True,
    ) -> dict[str, Any]:
        base = self.base_fate(tokens)
        base_action = base.get("action_fused_logits", base["action_logits"])
        base_reason = base["reason_logits"]
        label_tokens = base.get("label_tokens")
        if label_tokens is None:
            label_tokens = tokens.new_zeros(tokens.shape[0], self.action_dim + self.reason_dim, tokens.shape[-1])
        reason_targets = batch.get("reason") if batch is not None and "reason" in batch else None
        router = self.reason_router(base_action, base_reason, label_tokens, reason_targets=reason_targets, train_mode=self.training)
        active = router["active_reason_mask"]
        reason_tokens = label_tokens[:, self.action_dim : self.action_dim + self.reason_dim]
        expert_route = self.expert_router(reason_tokens, base_action, base_reason, active)
        use_structured = (self.training or force_structured_upper) and structured is not None
        bags = self.bag_builder.build(structured or [None] * tokens.shape[0], reason_targets, tokens.device) if use_structured else self.bag_builder.build([None] * tokens.shape[0], reason_targets, tokens.device)
        expert_tokens = []
        expert_scores = []
        expert_reliability = []
        expert_counts: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(EXPERT_TYPES):
            eout = self.experts[name](reason_tokens, tokens, bags["features"].get(name))
            expert_tokens.append(eout["evidence_tokens"])
            expert_scores.append(eout["evidence_scores"])
            expert_reliability.append(eout["evidence_reliability"])
            expert_counts[name] = eout["bag_count"]
        tok_stack = torch.stack(expert_tokens, dim=2)
        score_stack = torch.stack(expert_scores, dim=2)
        rel_stack = torch.stack(expert_reliability, dim=2)
        route_probs = expert_route["expert_route_probs"].unsqueeze(-1)
        evidence_tokens = (tok_stack * route_probs).sum(2)
        route_rel = (rel_stack * expert_route["expert_route_probs"]).sum(2)
        reason_up = self.reason_update(base_reason, reason_tokens, evidence_tokens, active, route_rel)
        action_fb = self.action_feedback(base_action, reason_up["reason_delta"], reason_up["reason_reliability"], epoch=epoch)
        route_mask = expert_route["expert_route_mask"]
        usage = {name: int(route_mask[..., i].sum().detach().cpu().item()) for i, name in enumerate(EXPERT_TYPES)}
        selected_score = (score_stack * expert_route["expert_route_probs"]).sum(2)
        random_score = score_stack.mean(2)
        out: dict[str, Any] = {
            **base,
            "action_base_logits": base_action,
            "reason_base_logits": base_reason,
            "reason_logits": reason_up["reason_logits"],
            "reason_final_candidate_logits": reason_up["reason_logits"],
            "action_final_candidate_logits": action_fb["action_final_candidate_logits"],
            "action_logits": action_fb["action_logits"],
            "action_guarded_logits": action_fb["action_logits"],
            "active_reason_scores": router["active_reason_scores"],
            "active_reason_mask": active,
            "active_reason_recall_train": router["active_reason_recall_train"],
            "expert_route_mask": route_mask,
            "expert_usage": usage,
            "evidence_scores": selected_score,
            "random_evidence_scores": random_score,
            "reason_reliability": reason_up["reason_reliability"],
            "action_gate": action_fb["action_gate"],
            "action_delta": action_fb["action_delta"],
            "reason_delta": reason_up["reason_delta"],
            "action_safe_state": action_fb["action_safe_state"],
            "diagnostics": {
                "active_reason_count_mean": float(active.sum(1).float().mean().detach().cpu().item()),
                "gt_positive_reason_coverage_train": float(router["active_reason_recall_train"].detach().cpu().item()) if self.training else 0.0,
                "expert_usage_by_type": usage,
                "top2_expert_violation_count": int((route_mask.sum(-1) > 2).sum().detach().cpu().item()),
                "selected_evidence_score_mean": float(selected_score.detach().mean().cpu().item()),
                "random_evidence_score_mean": float(random_score.detach().mean().cpu().item()),
                "selected_gt_random_drop_ratio": float((selected_score.detach().mean() - random_score.detach().mean()).cpu().item()),
                "action_residual_abs_max": float(action_fb["action_residual_abs_max"].detach().cpu().item()),
                "reason_residual_abs_max": float(reason_up["reason_delta"].detach().abs().max().cpu().item()),
                "action_safe_state": action_fb["action_safe_state"],
                "primary_test_uses_bdd100k_gt": bool((not self.training) and force_structured_upper),
            },
            "evidence_stats": {k: float(v.mean().detach().cpu().item()) for k, v in expert_counts.items()},
        }
        return out

