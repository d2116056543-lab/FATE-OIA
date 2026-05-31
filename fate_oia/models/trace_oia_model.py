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
    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        tail_labels: Sequence[int] = (12, 9, 5, 14, 6, 11, 10, 13),
        base_fate: FATEOIAFeatureModel | None = None,
        use_label_query: bool = True,
        enable_evidence_pooler: bool = True,
        enable_transport: bool = True,
        enable_label_corr: bool = True,
        action_final_mode: str = "base_only",
        token_compression: str = "none",
        max_evidence_units_per_image: int = 96,
        per_reason_topk_evidence: int = 8,
        reason_alpha_init: float = 0.08,
        reason_alpha_max_common: float = 0.24,
        reason_alpha_max_tail: float = 0.24,
        action_bias_init: float = 0.0,
        action_bias_max_abs: float = 1.0,
        safe_ensemble_init_base_weight: float = 0.90,
        safe_ensemble_max_r2a_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if action_final_mode not in {"base_only", "action_safe_selector"}:
            raise ValueError("TRACE mode requires action_final_mode='base_only' or 'action_safe_selector'.")
        if token_compression != "none":
            raise ValueError("TRACE primary mode requires token_compression='none'.")
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.action_final_mode = action_final_mode
        self.action_bias_max_abs = float(action_bias_max_abs)
        self.safe_ensemble_max_r2a_weight = float(safe_ensemble_max_r2a_weight)
        self.tail_labels = tuple(int(x) for x in tail_labels)
        self.base_fate = base_fate or FATEOIAFeatureModel(
            dim=dim,
            action_dim=action_dim,
            reason_dim=reason_dim,
            use_label_query=use_label_query,
        )
        self.evidence_pooler = (
            CausalEvidencePooler(
                dim=dim,
                max_evidence_units_per_image=max_evidence_units_per_image,
                per_reason_topk_evidence=per_reason_topk_evidence,
            )
            if enable_evidence_pooler
            else None
        )
        self.label_corr = EvidenceConditionedLabelCorr(dim=dim, reason_dim=reason_dim) if enable_label_corr else None
        self.transport = (
            ReasonCausalPrototypeTransport(TransportConfig(dim=dim, reason_dim=reason_dim, tail_labels=self.tail_labels))
            if enable_transport
            else None
        )
        self.reason_alpha = nn.Parameter(torch.full((reason_dim,), float(reason_alpha_init)))
        max_alpha = torch.full((reason_dim,), float(reason_alpha_max_common))
        for idx in self.tail_labels:
            if 0 <= idx < reason_dim:
                max_alpha[idx] = float(reason_alpha_max_tail)
        self.register_buffer("reason_alpha_max", max_alpha, persistent=False)
        self.register_buffer("reason_clip", torch.full((reason_dim,), 1.20), persistent=False)
        self.action_bias = nn.Parameter(torch.full((action_dim,), float(action_bias_init)))
        init_r2a = max(1e-4, min(float(safe_ensemble_max_r2a_weight), 1.0 - float(safe_ensemble_init_base_weight)))
        normalized = max(1e-4, min(1.0 - 1e-4, init_r2a / max(float(safe_ensemble_max_r2a_weight), 1e-6)))
        self.safe_ensemble_r2a_logit = nn.Parameter(torch.logit(torch.tensor(normalized)))

    def _empty_evidence(self, tokens: torch.Tensor) -> dict[str, Any]:
        b, _, d = tokens.shape
        return {
            "evidence_tokens": tokens.new_zeros((b, 1, d)),
            "evidence_mask": torch.ones((b, 1), dtype=torch.bool, device=tokens.device),
            "evidence_quality": tokens.new_ones((b, 1)),
            "evidence_source_type": torch.full((b, 1), 3, dtype=torch.long, device=tokens.device),
            "reason_evidence_mask": torch.zeros((b, self.reason_dim, 1), dtype=torch.bool, device=tokens.device),
            "real_evidence_mask": torch.zeros((b, 1), dtype=torch.bool, device=tokens.device),
            "reason_quality": tokens.new_zeros((b, self.reason_dim)),
            "counts": {"object": 0, "lane": 0, "drivable": 0, "fallback": 1},
            "meta": [[] for _ in range(b)],
            "key_match": [],
        }

    def _reason_alpha_eff(self) -> torch.Tensor:
        return torch.minimum(torch.clamp(self.reason_alpha, min=0.0), self.reason_alpha_max)

    def _final_reason_logits(self, base_reason: torch.Tensor, evidence_reason: torch.Tensor) -> torch.Tensor:
        alpha = self._reason_alpha_eff().to(base_reason.dtype)
        clipped = (evidence_reason - base_reason.detach()).clamp(
            -self.reason_clip.to(base_reason.dtype),
            self.reason_clip.to(base_reason.dtype),
        )
        return base_reason + alpha * clipped

    def _bounded_action_bias(self, dtype: torch.dtype) -> torch.Tensor:
        return (self.action_bias_max_abs * torch.tanh(self.action_bias)).to(dtype)

    def _safe_ensemble(self, base_plus_bias: torch.Tensor, reason_to_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w_r2a = self.safe_ensemble_max_r2a_weight * torch.sigmoid(self.safe_ensemble_r2a_logit)
        w_base = 1.0 - w_r2a
        prob = w_base * torch.sigmoid(base_plus_bias) + w_r2a * torch.sigmoid(reason_to_action)
        logits = torch.logit(prob.clamp(1e-4, 1.0 - 1e-4))
        return logits, torch.stack([w_base.detach(), w_r2a.detach()])

    def forward(
        self,
        tokens: torch.Tensor,
        batch: dict[str, Any] | None = None,
        grounding_cache: dict[str, dict[str, Any]] | None = None,
        epoch: int = 0,
        return_cf: bool = False,
        cf_targets: torch.Tensor | None = None,
        original_tokens: torch.Tensor | None = None,
        image_height: int = 360,
        image_width: int = 640,
        patch_size: int = 8,
        reason_rules: dict[int, set[str]] | None = None,
        max_positive_reasons_per_sample: int = 2,
        selected_action_mode: str | None = None,
    ) -> dict[str, Any]:
        base = self.base_fate(tokens)
        label_tokens = base.get("label_tokens")
        if label_tokens is None:
            raise RuntimeError("TraceOIAModel requires label-query label_tokens.")
        if self.evidence_pooler is not None:
            evidence = self.evidence_pooler(
                tokens=tokens,
                original_tokens=original_tokens if original_tokens is not None else tokens,
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
        reason_tokens = label_tokens[:, self.action_dim : self.action_dim + self.reason_dim]
        reliability = evidence.get("reason_quality", torch.zeros(reason_tokens.shape[:2], device=tokens.device, dtype=tokens.dtype))
        corr = self.label_corr(reason_tokens, reliability) if self.label_corr is not None else {}
        if corr:
            reason_tokens = corr["reason_tokens"]
        if self.transport is not None:
            transport = self.transport(reason_tokens, base["reason_logits"], evidence, self.tail_labels)
        else:
            transport = {
                "evidence_reason_logits": base["reason_logits"],
                "T_sparse_fraction": tokens.new_tensor(0.0),
                "transport_entropy": tokens.new_zeros(tokens.shape[0], self.reason_dim),
                "T": tokens.new_zeros(tokens.shape[0], self.reason_dim, 1),
                "source_mass_by_reason": tokens.new_zeros(tokens.shape[0], self.reason_dim, 4),
            }
        reason_logits = self._final_reason_logits(base["reason_logits"], transport["evidence_reason_logits"])
        action_base = base["action_logits"]
        action_bias_eff = self._bounded_action_bias(action_base.dtype)
        action_base_plus_bias = action_base + action_bias_eff
        reason_to_action = base.get("action_reason_logits", base.get("reason_to_action_logits", action_base))
        action_safe_ensemble, action_selector_weights = self._safe_ensemble(action_base_plus_bias, reason_to_action)
        candidates = {
            "base": action_base,
            "base_plus_bias": action_base_plus_bias,
            "reason_to_action": reason_to_action,
            "safe_ensemble": action_safe_ensemble,
        }
        if self.action_final_mode == "action_safe_selector":
            mode = selected_action_mode or "base_plus_bias"
            action_logits = candidates.get(mode, action_base_plus_bias)
        else:
            mode = "base"
            action_logits = action_base
        if return_cf and self.transport is not None:
            cf = transport_counterfactual_intervention(
                base["reason_logits"],
                transport["evidence_reason_logits"],
                transport,
                self.transport,
                cf_targets,
                self.tail_labels,
                self._reason_alpha_eff(),
                float(self.reason_clip.max().detach().cpu()),
                max_positive_reasons_per_sample,
            )
        else:
            cf = {}
        return {
            "action_logits": action_logits,
            "reason_logits": reason_logits,
            "base_action_logits": action_base,
            "base_action_visual_logits": base.get("action_visual_logits", action_base),
            "base_action_reason_logits": reason_to_action,
            "base_action_fused_logits": base.get("action_fused_logits", action_base),
            "base_reason_logits": base["reason_logits"],
            "action_visual_logits": base.get("action_visual_logits", action_base),
            "action_reason_logits": reason_to_action,
            "action_fused_logits": base.get("action_fused_logits", action_base),
            "reason_to_action_logits": reason_to_action,
            "action_logits_base_plus_bias": action_base_plus_bias,
            "action_logits_reason_to_action": reason_to_action,
            "action_logits_safe_ensemble": action_safe_ensemble,
            "action_logits_candidates": candidates,
            "action_candidates": candidates,
            "action_candidate_names": list(candidates.keys()),
            "action_selector_weights": action_selector_weights,
            "selected_action_mode": mode,
            "action_bias_eff": action_bias_eff,
            "reason_alpha_eff": self._reason_alpha_eff(),
            "label_tokens": label_tokens,
            "attention": base.get("attention"),
            "evidence": evidence,
            "transport": transport,
            "cf": cf,
            "diagnostics": {
                "action_protection_max_abs": float((action_logits - action_base).abs().max().detach().cpu()),
                "T_sparse_fraction": float(transport["T_sparse_fraction"].detach().cpu()) if torch.is_tensor(transport.get("T_sparse_fraction")) else float(transport.get("T_sparse_fraction", 0.0)),
            },
        }
