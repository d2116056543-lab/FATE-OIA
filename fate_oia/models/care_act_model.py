from __future__ import annotations

from typing import Any

import torch
from torch import nn

from fate_oia.models.care_action_evidence_experts import ActionEvidenceExpertBank
from fate_oia.models.care_action_set_head import ActionSetConsistencyHead
from fate_oia.models.care_moe_oia_model import CAREMoEOIAModel


def binary_entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    return (-(p * p.log() + (1.0 - p) * (1.0 - p).log()) / 0.69314718056).clamp(0.0, 1.0)


class CAREActOIAModel(nn.Module):
    def __init__(
        self,
        base_fate: nn.Module | None = None,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        test_top_k_reasons: int = 12,
        action_evidence_cap_max: float = 0.12,
        action_confident_cap: float = 0.02,
        action_set_cap: float = 0.06,
        reason_cap_common: float = 0.12,
        reason_cap_tail: float = 0.18,
        action_residual_warmup_epochs: int = 4,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.action_evidence_cap_max = action_evidence_cap_max
        self.action_confident_cap = action_confident_cap
        self.action_set_cap = action_set_cap
        self.action_residual_warmup_epochs = action_residual_warmup_epochs
        self.reason_model = CAREMoEOIAModel(
            base_fate=base_fate,
            dim=dim,
            action_dim=action_dim,
            reason_dim=reason_dim,
            test_top_k_reasons=test_top_k_reasons,
            action_cap_max=0.0,
            action_feedback_warmup_epochs=9999,
        )
        self.action_evidence_bank = ActionEvidenceExpertBank(dim=dim, action_dim=action_dim)
        self.action_set_head = ActionSetConsistencyHead(dim=dim, action_dim=action_dim, action_set_cap=action_set_cap)
        self.reason_gate = nn.Sequential(nn.Linear(reason_dim + action_dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, action_dim), nn.Sigmoid())

    def forward(
        self,
        tokens: torch.Tensor,
        batch: dict[str, Any] | None = None,
        structured: list[dict[str, Any] | None] | None = None,
        epoch: int = 0,
        force_structured_upper: bool = False,
        action_residual_shutdown: bool = False,
        return_diagnostics: bool = True,
    ) -> dict[str, Any]:
        reason_out = self.reason_model(tokens, batch=batch, structured=structured, epoch=epoch, force_structured_upper=force_structured_upper)
        base_action = reason_out["action_base_logits"]
        base_reason = reason_out["reason_base_logits"]
        reason_logits = reason_out["reason_logits"]
        label_tokens = reason_out.get("label_tokens")
        if label_tokens is None:
            label_tokens = tokens.new_zeros(tokens.shape[0], self.action_dim + self.reason_dim, tokens.shape[-1])
        action_tokens = label_tokens[:, : self.action_dim]
        reason_targets = batch.get("reason") if batch is not None and "reason" in batch else None
        use_structured = (self.training or force_structured_upper) and structured is not None
        bags = self.reason_model.bag_builder.build(structured or [None] * tokens.shape[0], reason_targets, tokens.device) if use_structured else None
        action_uncertainty = binary_entropy_from_logits(base_action)
        ev = self.action_evidence_bank(action_tokens, tokens, base_action, action_uncertainty, structured_bags=(bags or {}).get("features") if isinstance(bags, dict) else None, structured=structured)
        reason_reliability = reason_out["reason_reliability"]
        reason_gate = self.reason_gate(torch.cat([reason_reliability.detach(), action_uncertainty], dim=-1))
        dynamic_cap = torch.where(action_uncertainty < 0.35, torch.full_like(action_uncertainty, self.action_confident_cap), torch.full_like(action_uncertainty, self.action_evidence_cap_max))
        warm = 0.0 if epoch < self.action_residual_warmup_epochs or action_residual_shutdown else min(1.0, (epoch - self.action_residual_warmup_epochs + 1) / 4.0)
        action_evidence_delta = torch.tanh(ev["action_evidence_delta_raw"]) * dynamic_cap * reason_gate * warm
        action_evidence_logits = base_action + action_evidence_delta
        set_out = self.action_set_head(base_action, ev["action_evidence_context"], action_uncertainty)
        action_set_delta = set_out["action_set_delta"] * warm
        action_set_logits = base_action + action_set_delta
        action_total_delta = (action_evidence_delta + action_set_delta).clamp(-0.15, 0.15)
        action_candidate = base_action + action_total_delta
        out: dict[str, Any] = {
            **reason_out,
            "action_base_logits": base_action,
            "action_visual_logits": reason_out.get("action_visual_logits", base_action),
            "action_reason_logits": reason_out.get("action_reason_logits", reason_out.get("reason_to_action_logits", base_action)),
            "reason_to_action_logits": reason_out.get("reason_to_action_logits", reason_out.get("action_reason_logits", base_action)),
            "action_evidence_logits": action_evidence_logits,
            "action_set_logits": action_set_logits,
            "action_final_candidate_logits": action_candidate,
            "action_guarded_logits": action_candidate,
            "action_logits": action_candidate,
            "reason_base_logits": base_reason,
            "reason_logits": reason_logits,
            "reason_final_logits": reason_logits,
            "action_evidence_delta": action_evidence_delta,
            "action_set_delta": action_set_delta,
            "action_total_delta": action_total_delta,
            "action_uncertainty": action_uncertainty,
            "action_correction_gate": reason_gate,
            "action_branch_candidates": {
                "base": base_action,
                "evidence": action_evidence_logits,
                "action_set": action_set_logits,
                "candidate": action_candidate,
            },
            **set_out,
            **ev,
        }
        diag = dict(reason_out.get("diagnostics", {}))
        diag.update({
            "primary_test_uses_bdd100k_gt": bool((not self.training) and force_structured_upper),
            "action_evidence_delta_abs_max": float(action_evidence_delta.detach().abs().max().cpu().item()),
            "action_set_delta_abs_max": float(action_set_delta.detach().abs().max().cpu().item()),
            "action_total_delta_abs_max": float(action_total_delta.detach().abs().max().cpu().item()),
            "action_evidence_expert_usage": ev["action_expert_usage"],
            "selected_action_set_histogram": torch.bincount(set_out["selected_action_set_id"].detach().cpu(), minlength=self.action_set_head.action_set_prototypes.shape[0]).tolist(),
            "action_residual_warmup_active": bool(warm == 0.0),
            "action_residual_shutdown": bool(action_residual_shutdown),
        })
        out["diagnostics"] = diag
        return out
