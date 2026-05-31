from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from fate_oia.models.causal_evidence_pooler import CausalEvidencePooler
from fate_oia.models.evidence_conditioned_label_corr import EvidenceConditionedLabelCorr
from fate_oia.models.fate_oia_model import FATEOIAFeatureModel
from fate_oia.models.reason_causal_prototype_transport import ReasonCausalPrototypeTransport, TransportConfig
from fate_oia.models.transport_counterfactual import transport_counterfactual_intervention


class TraceOIAModel(nn.Module):
    def __init__(self, dim: int = 384, action_dim: int = 4, reason_dim: int = 21, tail_labels: Sequence[int] = (12, 9, 5, 14, 6, 11, 10, 13), base_fate: FATEOIAFeatureModel | None = None, use_label_query: bool = True, enable_evidence_pooler: bool = True, enable_transport: bool = True, enable_label_corr: bool = True, action_final_mode: str = "base_only", token_compression: str = "none", max_evidence_units_per_image: int = 96, per_reason_topk_evidence: int = 8) -> None:
        super().__init__()
        if action_final_mode != "base_only":
            raise ValueError("TRACE primary mode requires action_final_mode='base_only'.")
        if token_compression != "none":
            raise ValueError("TRACE primary mode requires token_compression='none'.")
        self.action_dim, self.reason_dim = int(action_dim), int(reason_dim)
        self.tail_labels = tuple(int(x) for x in tail_labels)
        self.base_fate = base_fate or FATEOIAFeatureModel(dim=dim, action_dim=action_dim, reason_dim=reason_dim, use_label_query=use_label_query)
        self.evidence_pooler = CausalEvidencePooler(dim=dim, max_evidence_units_per_image=max_evidence_units_per_image, per_reason_topk_evidence=per_reason_topk_evidence) if enable_evidence_pooler else None
        self.label_corr = EvidenceConditionedLabelCorr(dim=dim, reason_dim=reason_dim) if enable_label_corr else None
        self.transport = ReasonCausalPrototypeTransport(TransportConfig(dim=dim, reason_dim=reason_dim, tail_labels=self.tail_labels)) if enable_transport else None
        self.reason_alpha = nn.Parameter(torch.full((reason_dim,), 0.08))
        self.register_buffer("reason_alpha_max", torch.full((reason_dim,), 0.24), persistent=False)
        self.register_buffer("reason_clip", torch.full((reason_dim,), 1.20), persistent=False)

    def _empty_evidence(self, tokens: torch.Tensor) -> dict[str, Any]:
        b, _, d = tokens.shape
        return {"evidence_tokens": tokens.new_zeros((b, 1, d)), "evidence_mask": torch.ones((b, 1), dtype=torch.bool, device=tokens.device), "evidence_quality": tokens.new_ones((b, 1)), "evidence_source_type": torch.full((b, 1), 3, dtype=torch.long, device=tokens.device), "reason_evidence_mask": torch.zeros((b, self.reason_dim, 1), dtype=torch.bool, device=tokens.device), "real_evidence_mask": torch.zeros((b, 1), dtype=torch.bool, device=tokens.device), "reason_quality": tokens.new_zeros((b, self.reason_dim)), "counts": {"object": 0, "lane": 0, "drivable": 0, "fallback": 1}, "meta": [[] for _ in range(b)], "key_match": []}

    def _final_reason_logits(self, base_reason: torch.Tensor, evidence_reason: torch.Tensor) -> torch.Tensor:
        return base_reason + torch.minimum(torch.clamp(self.reason_alpha, min=0.0), self.reason_alpha_max).to(base_reason.dtype) * (evidence_reason - base_reason.detach()).clamp(-self.reason_clip.to(base_reason.dtype), self.reason_clip.to(base_reason.dtype))

    def forward(self, tokens: torch.Tensor, batch: dict[str, Any] | None = None, grounding_cache: dict[str, dict[str, Any]] | None = None, epoch: int = 0, return_cf: bool = False, cf_targets: torch.Tensor | None = None, original_tokens: torch.Tensor | None = None, image_height: int = 360, image_width: int = 640, patch_size: int = 8, reason_rules: dict[int, set[str]] | None = None, max_positive_reasons_per_sample: int = 2) -> dict[str, Any]:
        base = self.base_fate(tokens)
        label_tokens = base.get("label_tokens")
        if label_tokens is None:
            raise RuntimeError("TraceOIAModel requires label-query label_tokens.")
        evidence = self.evidence_pooler(tokens=tokens, original_tokens=original_tokens if original_tokens is not None else tokens, label_tokens=label_tokens, label_attention=base.get("attention"), batch=batch, grounding_cache=grounding_cache, image_height=image_height, image_width=image_width, patch_size=patch_size, reason_rules=reason_rules) if self.evidence_pooler is not None else self._empty_evidence(tokens)
        reason_tokens = label_tokens[:, self.action_dim : self.action_dim + self.reason_dim]
        preliminary_reliability = evidence.get("reason_quality", torch.zeros(reason_tokens.shape[:2], device=tokens.device, dtype=tokens.dtype))
        corr = self.label_corr(reason_tokens, preliminary_reliability) if self.label_corr is not None else {}
        if corr:
            reason_tokens = corr["reason_tokens"]
        transport = self.transport(reason_tokens, base["reason_logits"], evidence, self.tail_labels)
        reason_logits = self._final_reason_logits(base["reason_logits"], transport["evidence_reason_logits"])
        cf = transport_counterfactual_intervention(base["reason_logits"], transport["evidence_reason_logits"], transport, self.transport, cf_targets, self.tail_labels, torch.minimum(torch.clamp(self.reason_alpha, min=0.0), self.reason_alpha_max), float(self.reason_clip.max().detach().cpu()), max_positive_reasons_per_sample) if return_cf else {}
        return {"action_logits": base["action_logits"], "reason_logits": reason_logits, "base_action_logits": base["action_logits"], "base_reason_logits": base["reason_logits"], "action_visual_logits": base["action_visual_logits"], "action_reason_logits": base["action_reason_logits"], "action_fused_logits": base["action_logits"], "reason_to_action_logits": base["action_reason_logits"], "label_tokens": label_tokens, "attention": base.get("attention"), "evidence": evidence, "transport": transport, "cf": cf, "diagnostics": {"action_protection_max_abs": float((base["action_logits"] - base["action_logits"]).abs().max().detach().cpu()), "T_sparse_fraction": float(transport["T_sparse_fraction"].detach().cpu())}}
