from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .acpr_dino_field import ACPRDinoFieldExtractor
from .acpr_ego_regions import ACPREgoRegionEncoder
from .acpr_predicate_reason import ACPRPredicateReasoner
from .acpr_scene_predicate_head import ACPRScenePredicateHead
from .aie_contribution_head import AIEContributionHead
from .aie_evidence_interface import AIEEvidenceInterface
from .pact_context_decoder import PACTContextDecoder
from .pact_explanation_decoder import PACTExplanationDecoder
from .pact_predicate_agreement import PACTPredicateAgreement
from .pact_reason_rereader import PACTReasonRereader
from .pact_shared_readout import PACTSharedVisualReadout, licensed_gradient


class PACTOIAModel(nn.Module):
    """Pareto-licensed role split with one frozen visual backbone call."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        scene_config: str = "configs/aie_scene_predicates.yaml",
        grammar_path: str = "configs/acpr_reason_predicate_grammar.yaml",
        use_mock_dino: bool = False,
        mock_dim: int | None = None,
        probes_per_action: int = 4,
        local_points_per_layer: int = 8,
        max_offset: float = 0.25,
        predicate_bias_max: float = 0.25,
        probe_chunk_size: int = 16,
        action_kappa: float = 3.0,
        action_logit_norm_cap: float = 20.0,
        reason_kappa: float = 4.0,
    ) -> None:
        super().__init__()
        self.action_dim, self.reason_dim = int(action_dim), int(reason_dim)
        self.dino = ACPRDinoFieldExtractor(
            selected_layers=selected_layers, pretrained_weights=pretrained_weights,
            use_mock_dino=use_mock_dino, mock_dim=mock_dim or dim,
        )
        self.ego = ACPREgoRegionEncoder(grid_hw=(45, 80), dim=dim)
        self.predicate_head = ACPRScenePredicateHead(scene_config=scene_config, dim=dim, num_layers=len(selected_layers))
        self.shared_readout = PACTSharedVisualReadout(dim, action_dim, reason_dim)
        self.context_decoder = PACTContextDecoder(dim, action_dim, reason_dim)
        self.explanation_decoder = PACTExplanationDecoder(dim, action_dim, reason_dim)
        self.predicate_reason = ACPRPredicateReasoner(
            dim, reason_dim, self.predicate_head.num_predicates, self.predicate_head.names, grammar_path
        )
        self.action_evidence = AIEEvidenceInterface(
            dim, action_dim, probes_per_action, len(selected_layers), self.predicate_head.num_predicates,
            (45, 80), local_points_per_layer, max_offset, predicate_bias_max, probe_chunk_size,
        )
        self.predicate_agreement = PACTPredicateAgreement(predicate_bias_max)
        self.action_contribution = AIEContributionHead(
            dim, action_dim, probes_per_action, action_kappa, action_logit_norm_cap
        )
        self.reason_private = PACTReasonRereader(
            dim, reason_dim, action_dim, probes_per_action, self.predicate_head.num_predicates,
            self.predicate_head.names, grammar_path, len(selected_layers), kappa=reason_kappa,
        )

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        return self.dino(images)

    def decode_from_field(
        self,
        field: dict[str, Any],
        *,
        semantic_share_license: float = 0.0,
        action_scale: float = 0.0,
        reason_budget: float = 0.0,
        compatibility_mode: bool = False,
        predicate_bias_enabled: bool = True,
    ) -> dict[str, Any]:
        raw_patch = field["patch_tokens_by_layer"]
        patch0, ego_features, region_masks, ego_stats = self.ego(raw_patch[:, 0])
        patch = raw_patch.clone()
        patch[:, 0] = patch0

        # Both views are numerically identical. Only the semantic view blocks visual gradients.
        action_predicate = self.predicate_head(patch, region_masks=region_masks)
        semantic_predicate = self.predicate_head(patch.detach(), region_masks=region_masks)
        shared = self.shared_readout(patch)
        context = self.context_decoder(shared["shared_label_nodes"], action_predicate["predicate_tokens"])
        explanation_input = licensed_gradient(shared["shared_label_nodes"], semantic_share_license)
        explanation = self.explanation_decoder(explanation_input, semantic_predicate["predicate_tokens"])
        reason_predicate = self.predicate_reason(
            explanation["reason_nodes_formal"],
            semantic_predicate["predicate_probs"],
            semantic_predicate["predicate_tokens"],
        )
        reason_primary = explanation["reason_logits_visual_formal"] + reason_predicate["predicate_reason_delta"]
        evidence = self.action_evidence(
            context["action_nodes_context"], raw_patch,
            action_predicate["predicate_attention"], action_predicate["predicate_probs"],
            predicate_bias_enabled=predicate_bias_enabled,
            predicate_agreement=self.predicate_agreement,
            predicate_agreement_bypass=compatibility_mode,
        )
        mixture_quality, mixture_id = evidence["predicate_compatibility"].max(-1)
        named = (evidence.get("predicate_visual_agreement", mixture_quality) *
                 evidence.get("predicate_confidence", mixture_quality)) >= 0.05
        formal_name_id = torch.where(named, mixture_id, torch.full_like(mixture_id, -1))
        contribution = self.action_contribution(
            evidence["evidence_token"], context["action_logits_primary"], action_scale=action_scale
        )
        reason = self.reason_private(
            explanation["reason_nodes_formal"], raw_patch, evidence["evidence_token"], evidence["evidence_map"],
            contribution["bounded_contribution"], semantic_predicate["predicate_attention"],
            semantic_predicate["predicate_probs"], reason_primary,
            reason_budget=reason_budget, compatibility_mode=compatibility_mode,
            reason_scale=reason_budget if compatibility_mode else 1.0,
        )
        return {
            **field, **shared, **context, **explanation, **action_predicate,
            **reason_predicate, **evidence, **contribution, **reason,
            "patch_tokens_by_layer_raw": raw_patch,
            "patch_tokens_by_layer_ego": patch,
            "ego_features": ego_features, "ego_region_masks": region_masks, "ego_stats": ego_stats,
            "action_predicate_logits": action_predicate["predicate_logits"],
            "semantic_predicate_logits": semantic_predicate["predicate_logits"],
            "semantic_predicate_probs": semantic_predicate["predicate_probs"],
            "semantic_predicate_attention": semantic_predicate["predicate_attention"],
            "semantic_predicate_tokens": semantic_predicate["predicate_tokens"],
            "formal_predicate_name_id": formal_name_id,
            "formal_predicate_name_quality": mixture_quality,
            "_grammar_positive_mask": self.predicate_reason.positive_mask,
            "_grammar_contradictory_mask": self.predicate_reason.contradictory_mask,
            "action_nodes_primary": context["action_nodes_context"],
            "reason_nodes_primary": explanation["reason_nodes_formal"],
            "action_visual_logits_primary": context["action_visual_logits"],
            "action_reason_logits_primary": context["action_context_logits"],
            "action_fusion_gate_primary": context["action_fusion_gate"],
            "reason_logits_visual_primary": explanation["reason_logits_visual_formal"],
            "predicate_reason_delta_primary": reason_predicate["predicate_reason_delta"],
            "reason_logits_primary": reason_primary,
            "branch_logits": {
                "primary_action": context["action_logits_primary"],
                "final_action": contribution["action_logits_final"],
                "primary_reason": reason_primary,
                "final_reason": reason["reason_logits_final"],
            },
        }

    def forward(self, images: Tensor, **kwargs) -> dict[str, Any]:
        return self.decode_from_field(self.encode_images(images), **kwargs)

    def rerun_action_evidence_from_field(
        self, modified_field: dict[str, Any], fixed_primary: dict[str, Tensor], *,
        action_scale: float, predicate_bias_enabled: bool,
    ) -> dict[str, Tensor]:
        evidence = self.action_evidence(
            fixed_primary["action_nodes_primary"].detach(), modified_field["patch_tokens_by_layer_raw"],
            fixed_primary["predicate_attention"].detach(), fixed_primary["predicate_probs"].detach(),
            predicate_bias_enabled=predicate_bias_enabled, predicate_agreement=self.predicate_agreement,
            predicate_agreement_bypass=False,
        )
        contribution = self.action_contribution(
            evidence["evidence_token"], fixed_primary["action_logits_primary"].detach(), action_scale=action_scale
        )
        return {**evidence, **contribution}

    @torch.no_grad()
    def migrate_from_aie_state_dict(self, state: Mapping[str, Tensor], strict: bool = True) -> dict[str, list[str]]:
        mapped: dict[str, Tensor] = {}
        direct = ("dino", "ego", "predicate_head", "predicate_reason", "action_evidence", "action_contribution", "reason_private")
        for key, value in state.items():
            for name in direct:
                prefix = f"foundation.{name}." if name in {"dino", "ego", "predicate_head", "predicate_reason"} else f"{name}."
                if key.startswith(prefix):
                    mapped[f"{name}.{key[len(prefix):]}"] = value
            trunk_prefix = "foundation.trunk."
            if key.startswith(trunk_prefix):
                suffix = key[len(trunk_prefix):]
                if suffix.startswith(("label_queries", "key_proj.", "value_proj.", "query_proj.")):
                    mapped[f"shared_readout.{suffix}"] = value
                if suffix.startswith(("label_self_attn.", "predicate_cross_attn.", "predicate_gate", "logit_head.")):
                    mapped[f"context_decoder.{suffix}"] = value
                    mapped[f"explanation_decoder.{suffix}"] = value
                if suffix.startswith(("reason_to_action.", "action_visual_head.", "fusion_gate.")):
                    mapped[f"context_decoder.{suffix}"] = value
        result = self.load_state_dict(mapped, strict=False)
        allowed_missing = {"predicate_agreement.learned_gate"}
        missing = [key for key in result.missing_keys if key not in allowed_missing]
        if strict and (missing or result.unexpected_keys):
            raise RuntimeError(f"PACT migration failed: missing={missing}, unexpected={result.unexpected_keys}")
        return {"missing_keys": missing, "unexpected_keys": list(result.unexpected_keys)}
