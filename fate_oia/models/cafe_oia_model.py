from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.causal_evidence_pooler import CausalEvidencePooler
from fate_oia.models.causal_label_grouping import CausalLabelGrouping
from fate_oia.models.evidence_gated_residual import EvidenceGatedActionResidual, EvidenceGatedReasonResidual
from fate_oia.models.semantic_shapley_lite import SemanticShapleyLite


class CAFEOIAModel(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        use_label_query: bool = True,
        tail_labels: Sequence[int] = (12, 9, 5, 14, 6, 11, 10, 13),
        base_fate: FATEOIAFeatureModel | None = None,
        enable_evidence_pooler: bool = True,
        enable_causal_label_grouping: bool = True,
        enable_visual_residual: bool = True,
        enable_action_residual: bool = True,
        enable_shapley_lite: bool = True,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.tail_labels = tuple(int(x) for x in tail_labels)
        self.base_fate = base_fate or FATEOIAFeatureModel(dim=dim, action_dim=action_dim, reason_dim=reason_dim, use_label_query=use_label_query)
        self.evidence_pooler = CausalEvidencePooler(dim=dim) if enable_evidence_pooler else None
        self.causal_label_grouping = CausalLabelGrouping(dim=dim) if enable_causal_label_grouping else None
        self.visual_residual = EvidenceGatedReasonResidual(dim=dim, reason_dim=reason_dim, tail_labels=tuple(self.tail_labels)) if enable_visual_residual else None
        self.action_residual = EvidenceGatedActionResidual(reason_dim=reason_dim, action_dim=action_dim) if enable_action_residual else None
        self.shapley_lite = SemanticShapleyLite() if enable_shapley_lite else None
        self.label_group_scale = nn.Parameter(torch.tensor(0.03))
        self.label_group_scale_max = 0.10

    def forward(
        self,
        tokens: torch.Tensor,
        batch: dict[str, Any] | None = None,
        grounding_cache: dict[str, dict[str, Any]] | None = None,
        epoch: int = 0,
        return_cf: bool = False,
        cf_targets: torch.Tensor | None = None,
        cf_mode: str = "none",
        original_tokens: torch.Tensor | None = None,
        provenance: torch.Tensor | None = None,
        image_height: int = 360,
        image_width: int = 640,
        patch_size: int = 8,
        reason_rules: dict[int, set[str]] | None = None,
    ) -> dict[str, Any]:
        base = self.base_fate(tokens)
        label_tokens = base.get("label_tokens")
        if label_tokens is None:
            raise RuntimeError("CAFEOIAModel requires base_fate label_tokens; use_label_query must be enabled.")
        if not hasattr(self.base_fate, "label_head") or not hasattr(self.base_fate.label_head, "cls"):
            raise RuntimeError("CAFEOIAModel requires base_fate.label_head.cls for causal label logits.")
        if self.evidence_pooler is not None:
            evidence = self.evidence_pooler(
                tokens=tokens,
                original_tokens=original_tokens,
                label_tokens=label_tokens,
                label_attention=base.get("attention"),
                batch=batch,
                grounding_cache=grounding_cache,
                image_height=image_height,
                image_width=image_width,
                patch_size=patch_size,
                reason_rules=reason_rules,
            )
        else:
            b = tokens.shape[0]
            evidence = {
                "evidence_tokens": tokens.new_zeros((b, 1, tokens.shape[-1])),
                "evidence_mask": torch.zeros((b, 1), dtype=torch.bool, device=tokens.device),
                "evidence_quality": tokens.new_zeros((b, 1)),
                "reason_quality": tokens.new_zeros((b, self.reason_dim)),
                "counts": {"object": 0, "lane": 0, "drivable": 0, "fallback": 0},
                "meta": [],
            }
        reason_tokens = label_tokens[:, self.action_dim : self.action_dim + self.reason_dim]
        if self.causal_label_grouping is not None:
            grouped = self.causal_label_grouping(reason_tokens, base["reason_logits"], evidence.get("reason_quality"), self.training)
        else:
            grouped = reason_tokens
        causal_reason_logits = self.base_fate.label_head.cls(grouped).squeeze(-1)
        scale = torch.clamp(self.label_group_scale, 0.0, self.label_group_scale_max)
        if self.visual_residual is not None:
            residual = self.visual_residual(grouped, base["reason_logits"], causal_reason_logits, evidence)
            reason_delta = residual["reason_delta"]
            reason_gate = residual["reason_gate"]
        else:
            reason_delta = torch.zeros_like(base["reason_logits"])
            reason_gate = torch.zeros_like(base["reason_logits"])
        reason_logits = base["reason_logits"] + scale * (causal_reason_logits - base["reason_logits"]) + reason_gate * reason_delta
        action_reason_logits = self.base_fate.reason_to_action(reason_logits)
        if self.action_residual is not None:
            action_out = self.action_residual(base["action_logits"], action_reason_logits, evidence, reason_logits, base["reason_logits"])
            action_delta = action_out["action_delta"]
            action_beta = action_out["action_beta"]
        else:
            action_delta = torch.zeros_like(action_reason_logits)
            action_beta = torch.zeros_like(action_reason_logits)
        action_logits = action_reason_logits + action_delta
        cf = {}
        if return_cf:
            target_deleted_reason = reason_logits - torch.relu(reason_gate * reason_delta)
            context_only_reason = base["reason_logits"]
            evidence_only_reason = reason_gate * reason_delta
            cf = {
                "reason_logits_factual": reason_logits,
                "reason_logits_target_deleted": target_deleted_reason,
                "reason_logits_context_only": context_only_reason,
                "reason_logits_evidence_only": evidence_only_reason,
                "reason_logits_replaced": context_only_reason - torch.relu(reason_gate * reason_delta),
                "action_logits_factual": action_logits,
                "action_logits_target_deleted": self.base_fate.reason_to_action(target_deleted_reason),
                "action_logits_context_only": base["action_logits"],
                "action_logits_evidence_only": self.base_fate.reason_to_action(evidence_only_reason),
            }
        shapley = self.shapley_lite(reason_logits, base["reason_logits"], evidence["evidence_quality"]) if self.shapley_lite is not None else {}
        return {
            "action_logits": action_logits,
            "reason_logits": reason_logits,
            "action_visual_logits": base["action_visual_logits"],
            "action_reason_logits": action_reason_logits,
            "action_fused_logits": action_logits,
            "reason_to_action_logits": action_reason_logits,
            "base_action_logits": base["action_logits"],
            "base_reason_logits": base["reason_logits"],
            "causal_reason_logits": causal_reason_logits,
            "reason_delta_logits": reason_delta,
            "reason_gate": reason_gate,
            "action_delta_logits": action_delta,
            "action_beta": action_beta,
            "fusion_gate": base["fusion_gate"],
            "label_tokens": label_tokens,
            "attention": base.get("attention"),
            "evidence": evidence,
            "cf": cf,
            "diagnostics": {
                "label_group_scale": float(scale.detach().cpu()),
                "reason_gate_mean": float(reason_gate.detach().mean().cpu()),
                "action_beta_mean": float(action_beta.detach().mean().cpu()),
                **{k: float(v.detach().mean().cpu()) for k, v in shapley.items() if isinstance(v, torch.Tensor)},
            },
        }

