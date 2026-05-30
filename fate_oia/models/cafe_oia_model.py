from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.causal_evidence_pooler import CausalEvidencePooler
from fate_oia.models.causal_label_grouping import CausalLabelGrouping
from fate_oia.models.counterfactual_evidence_intervention import (
    make_evidence_override,
    select_positive_reasons,
    target_unit_mask,
)
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
        enable_action_residual: bool = False,
        enable_shapley_lite: bool = True,
        evidence_pooler_version: str = "v2",
        max_evidence_units_per_image: int = 96,
        per_reason_topk_evidence: int = 8,
        fallback_quality_multiplier: float = 0.20,
        action_update_scale_init: float = 0.03,
        action_update_scale_max: float = 0.10,
        allow_fallback_counterfactual: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.tail_labels = tuple(int(x) for x in tail_labels)
        self.base_fate = base_fate or FATEOIAFeatureModel(dim=dim, action_dim=action_dim, reason_dim=reason_dim, use_label_query=use_label_query)
        self.evidence_pooler = (
            CausalEvidencePooler(
                dim=dim,
                max_evidence_units_per_image=max_evidence_units_per_image,
                per_reason_topk_evidence=per_reason_topk_evidence,
                fallback_quality_multiplier=fallback_quality_multiplier,
                evidence_pooler_version=evidence_pooler_version,
            )
            if enable_evidence_pooler
            else None
        )
        self.causal_label_grouping = CausalLabelGrouping(dim=dim) if enable_causal_label_grouping else None
        self.visual_residual = EvidenceGatedReasonResidual(dim=dim, reason_dim=reason_dim, tail_labels=tuple(self.tail_labels)) if enable_visual_residual else None
        self.action_residual = EvidenceGatedActionResidual(reason_dim=reason_dim, action_dim=action_dim) if enable_action_residual else None
        self.shapley_lite = SemanticShapleyLite() if enable_shapley_lite else None
        self.label_group_scale = nn.Parameter(torch.tensor(0.03))
        self.label_group_scale_max = 0.10
        self.action_update_scale = nn.Parameter(torch.tensor(float(action_update_scale_init)))
        self.action_update_scale_max = float(action_update_scale_max)
        self.null_evidence_token = nn.Parameter(torch.zeros(dim))
        self.allow_fallback_counterfactual = bool(allow_fallback_counterfactual)

    def _empty_evidence(self, tokens: torch.Tensor) -> dict[str, Any]:
        b, _, d = tokens.shape
        return {
            "evidence_tokens": tokens.new_zeros((b, 1, d)),
            "evidence_mask": torch.zeros((b, 1), dtype=torch.bool, device=tokens.device),
            "evidence_quality": tokens.new_zeros((b, 1)),
            "evidence_source_type": torch.full((b, 1), 3, dtype=torch.long, device=tokens.device),
            "evidence_patch_mask": torch.zeros((b, 1, max(1, tokens.shape[1] - 1)), dtype=torch.bool, device=tokens.device),
            "reason_evidence_mask": torch.zeros((b, self.reason_dim, 1), dtype=torch.bool, device=tokens.device),
            "real_evidence_mask": torch.zeros((b, 1), dtype=torch.bool, device=tokens.device),
            "reason_quality": tokens.new_zeros((b, self.reason_dim)),
            "counts": {"object": 0, "lane": 0, "drivable": 0, "fallback": 0},
            "meta": [],
            "key_match": [],
        }

    def forward_from_base_and_evidence(
        self,
        base: dict[str, torch.Tensor],
        evidence: dict[str, Any],
        mode: str = "factual",
    ) -> dict[str, torch.Tensor]:
        label_tokens = base.get("label_tokens")
        if label_tokens is None:
            raise RuntimeError("CAFEOIAModel requires base_fate label_tokens; use_label_query must be enabled.")
        if not hasattr(self.base_fate, "label_head") or not hasattr(self.base_fate.label_head, "cls"):
            raise RuntimeError("CAFEOIAModel requires base_fate.label_head.cls for causal label logits.")
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
            action_logits = action_reason_logits + action_delta
        else:
            action_update = torch.tanh(action_reason_logits - base["action_logits"])
            action_scale = torch.clamp(self.action_update_scale, 0.0, self.action_update_scale_max)
            action_delta = action_scale * action_update
            action_beta = action_scale.expand_as(action_delta)
            action_logits = base["action_logits"] + action_delta
        return {
            "action_logits": action_logits,
            "reason_logits": reason_logits,
            "action_reason_logits": action_reason_logits,
            "causal_reason_logits": causal_reason_logits,
            "reason_delta_logits": reason_delta,
            "reason_gate": reason_gate,
            "action_delta_logits": action_delta,
            "action_beta": action_beta,
            "mode": mode,
        }

    def forward_counterfactuals(
        self,
        base: dict[str, torch.Tensor],
        evidence: dict[str, Any],
        factual: dict[str, torch.Tensor],
        reason_labels: torch.Tensor | None = None,
        max_positive_reasons_per_sample: int = 2,
    ) -> dict[str, Any]:
        selected = select_positive_reasons(reason_labels, factual["reason_logits"], self.tail_labels, max_positive_reasons_per_sample)
        target_mask = target_unit_mask(evidence, selected, allow_fallback=self.allow_fallback_counterfactual)
        deleted_evidence = make_evidence_override(evidence, target_mask, "target_deleted", self.null_evidence_token)
        context_evidence = make_evidence_override(evidence, target_mask, "context_only", self.null_evidence_token)
        evidence_only = make_evidence_override(evidence, target_mask, "evidence_only", self.null_evidence_token)
        replaced_evidence = make_evidence_override(evidence, target_mask, "replaced", self.null_evidence_token)
        deleted = self.forward_from_base_and_evidence(base, deleted_evidence, mode="target_deleted")
        context = self.forward_from_base_and_evidence(base, context_evidence, mode="context_only")
        only = self.forward_from_base_and_evidence(base, evidence_only, mode="evidence_only")
        replaced = self.forward_from_base_and_evidence(base, replaced_evidence, mode="replaced")
        real = evidence.get("real_evidence_mask")
        if isinstance(real, torch.Tensor):
            real_target = target_mask & real
        else:
            real_target = target_mask
        valid = torch.zeros_like(factual["reason_logits"], dtype=torch.bool)
        counts = torch.zeros_like(factual["reason_logits"])
        reason_ev = evidence.get("reason_evidence_mask")
        if isinstance(reason_ev, torch.Tensor):
            for b, reasons in enumerate(selected):
                for r in reasons:
                    if 0 <= int(r) < valid.shape[1]:
                        mask = reason_ev[b, int(r)] & real_target[b]
                        valid[b, int(r)] = bool(mask.any())
                        counts[b, int(r)] = mask.float().sum()
        return {
            "reason_logits_factual": factual["reason_logits"],
            "reason_logits_target_deleted": deleted["reason_logits"],
            "reason_logits_context_only": context["reason_logits"],
            "reason_logits_evidence_only": only["reason_logits"],
            "reason_logits_replaced": replaced["reason_logits"],
            "action_logits_factual": factual["action_logits"],
            "action_logits_target_deleted": deleted["action_logits"],
            "action_logits_context_only": context["action_logits"],
            "action_logits_evidence_only": only["action_logits"],
            "action_logits_replaced": replaced["action_logits"],
            "cf_valid_mask": valid,
            "cf_real_evidence_mask": valid,
            "cf_target_evidence_count": counts,
            "cf_selected_reason_indices": selected,
            "cf_target_unit_mask": target_mask,
            "cf_intervention_type": "evidence_unit_mask",
            "cf_is_proxy": False,
        }

    def forward(
        self,
        tokens: torch.Tensor,
        batch: dict[str, Any] | None = None,
        grounding_cache: dict[str, dict[str, Any]] | None = None,
        epoch: int = 0,
        return_cf: bool = False,
        cf_targets: torch.Tensor | None = None,
        cf_mode: str = "evidence_unit_intervention",
        original_tokens: torch.Tensor | None = None,
        provenance: torch.Tensor | None = None,
        image_height: int = 360,
        image_width: int = 640,
        patch_size: int = 8,
        reason_rules: dict[int, set[str]] | None = None,
        max_positive_reasons_per_sample: int = 2,
    ) -> dict[str, Any]:
        base = self.base_fate(tokens)
        label_tokens = base.get("label_tokens")
        if label_tokens is None:
            raise RuntimeError("CAFEOIAModel requires label-query label_tokens.")
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
            evidence = self._empty_evidence(tokens)
        factual = self.forward_from_base_and_evidence(base, evidence, mode="factual")
        cf: dict[str, Any] = {}
        if return_cf:
            cf = self.forward_counterfactuals(
                base,
                evidence,
                factual,
                reason_labels=cf_targets,
                max_positive_reasons_per_sample=max_positive_reasons_per_sample,
            )
            if cf.get("cf_is_proxy"):
                raise RuntimeError("CAFE-OIA V2 forbids proxy counterfactual outputs.")
        shapley = self.shapley_lite(factual["reason_logits"], base["reason_logits"], evidence["evidence_quality"]) if self.shapley_lite is not None else {}
        return {
            "action_logits": factual["action_logits"],
            "reason_logits": factual["reason_logits"],
            "action_visual_logits": base["action_visual_logits"],
            "action_reason_logits": factual["action_reason_logits"],
            "action_fused_logits": factual["action_logits"],
            "reason_to_action_logits": factual["action_reason_logits"],
            "base_action_logits": base["action_logits"],
            "base_reason_logits": base["reason_logits"],
            "causal_reason_logits": factual["causal_reason_logits"],
            "reason_delta_logits": factual["reason_delta_logits"],
            "reason_gate": factual["reason_gate"],
            "action_delta_logits": factual["action_delta_logits"],
            "action_beta": factual["action_beta"],
            "fusion_gate": base["fusion_gate"],
            "label_tokens": label_tokens,
            "attention": base.get("attention"),
            "evidence": evidence,
            "cf": cf,
            "diagnostics": {
                "label_group_scale": float(torch.clamp(self.label_group_scale, 0.0, self.label_group_scale_max).detach().cpu()),
                "reason_gate_mean": float(factual["reason_gate"].detach().mean().cpu()),
                "action_beta_mean": float(factual["action_beta"].detach().mean().cpu()),
                **{k: float(v.detach().mean().cpu()) for k, v in shapley.items() if isinstance(v, torch.Tensor)},
            },
        }
