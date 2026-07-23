from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import torch
from torch import nn

from fate_oia.models.acpr_threshold_head import ACPRThresholdHead
from fate_oia.models.precise_annotation_head import PRECISEAnnotationHead
from fate_oia.models.precise_category_decoder import PRECISECategoryDecoder
from fate_oia.models.precise_dino_field import PRECISEDinoFieldExtractor
from fate_oia.models.precise_evidence_fields import PRECISEEvidenceFields
from fate_oia.models.precise_semantic_exchange import PRECISESemanticExchange
from fate_oia.models.precise_visual_field import PRECISEVisualField, VisualFieldBundle
from fate_oia.models.precise_visual_rereader import PRECISEVisualRereader
from fate_oia.utils.precise_schema import load_action_semantics, load_evidence_fields, load_reason_semantics


class PRECISEOIAModel(nn.Module):
    def __init__(self, config_root: str | Path = "configs", pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth", use_mock_dino: bool = False, evidence_schema: list[dict[str, Any]] | None = None, model_config: dict[str, Any] | None = None) -> None:
        super().__init__()
        root = Path(config_root)
        config = model_config or {}
        backbone = config.get("backbone", {})
        visual = config.get("visual", {})
        category = config.get("category", {})
        evidence_config = config.get("evidence", {})
        reason = config.get("reason", {})
        exchange = config.get("exchange", {})
        intervention = config.get("intervention", {})
        if backbone and (not backbone.get("freeze_backbone", True) or not backbone.get("no_grad_backbone", True)):
            raise ValueError("PRECISE requires a frozen no-grad DINO backbone")
        if int(category.get("intra_layers", 1)) != 1 or int(category.get("reread_passes", 1)) != 1:
            raise ValueError("PRECISE V1 requires one intra-category layer and one reread pass")
        dim = int(visual.get("dim", 384))
        if dim != 384:
            raise ValueError("PRECISE V1 official DINO field dimension must remain 384")
        self.reason_schema = load_reason_semantics(root / "precise_reason_semantics.yaml")
        self.action_schema = load_action_semantics(root / "precise_action_semantics.yaml")
        self.evidence_schema = evidence_schema if evidence_schema is not None else load_evidence_fields(root / "precise_evidence_fields.yaml")
        expected_parts = {"traffic_control": int(evidence_config.get("traffic_parts", 4)), "actor": int(evidence_config.get("actor_parts", 4)), "drivable": int(evidence_config.get("region_parts", 8)), "boundary": int(evidence_config.get("curve_parts", 8))}
        if any(int(field["num_parts"]) != expected_parts[field["family"]] for field in self.evidence_schema):
            raise ValueError("PRECISE active evidence schema does not match configured part counts")
        self.dino = PRECISEDinoFieldExtractor(
            arch=str(backbone.get("arch", "vit_small")),
            patch_size=int(backbone.get("patch_size", config.get("patch_size", 8))),
            selected_layers=tuple(int(value) for value in backbone.get("selected_layers", (3, 7, 11))),
            pretrained_weights=pretrained_weights,
            use_mock_dino=use_mock_dino,
        )
        self.visual_field = PRECISEVisualField(dim=dim, hidden=int(visual.get("adapter_hidden", 192)), local_kernel=int(visual.get("local_kernel", 3)), rezero_init=float(visual.get("rezero_init", 0.02)), context_pool_hw=tuple(visual.get("context_pool_hw", (9, 16))))
        self.category_decoder = PRECISECategoryDecoder(self.reason_schema, self.action_schema, dim=dim, heads=int(category.get("heads", 4)))
        self.evidence_fields = PRECISEEvidenceFields(self.evidence_schema, dim=dim, latent_slots=int(evidence_config.get("latent_slots", 6)), latent_parts=int(evidence_config.get("latent_parts", 4)), reliability_tau=float(evidence_config.get("reliability_tau", 0.20)))
        self.rereader = PRECISEVisualRereader(dim=dim, sampling_points_per_layer=int(category.get("sampling_points_per_layer", 4)), gamma_init=float(category.get("reread_gamma_init", 0.08)), gamma_max=float(category.get("reread_gamma_max", 0.35)))
        self.exchange = PRECISESemanticExchange(self.evidence_schema, self.reason_schema, dim=dim, overlap_tau=float(exchange.get("overlap_tau", 0.08)), overlap_slope=float(exchange.get("overlap_slope", 12.0)), reliability_eps=float(exchange.get("reliability_eps", 1e-4)), action_gamma_init=float(category.get("action_exchange_gamma_init", 0.05)), reason_gamma_init=float(category.get("reason_exchange_gamma_init", 0.05)), gamma_max=float(category.get("exchange_gamma_max", 0.25)))
        self.annotation_head = PRECISEAnnotationHead(dim=dim, rank=int(reason.get("annotation_rank", 8)), delta_cap=float(reason.get("annotation_delta_cap", 0.75)))
        self.semantic_weight_floor = float(reason.get("semantic_weight_floor", 0.25))
        self.intervention_margin = float(intervention.get("margin", 0.10))
        self.intervention_control_mass_tolerance = float(intervention.get("control_mass_tolerance", 0.05))
        self.threshold_head = ACPRThresholdHead(action_dim=4, reason_dim=21)
        self.action_refined_head = nn.Linear(384, 1)
        self.reason_refined_head = nn.Linear(384, 1)
        self.reason_latent_query = nn.Linear(384, 384, bias=False)
        self.reason_latent_value = nn.Linear(384, 384, bias=False)
        self.reason_latent_gamma_raw = nn.Parameter(torch.tensor(-1.3862944))

    def encode_images(self, images: torch.Tensor) -> VisualFieldBundle:
        return self.visual_field(self.dino(images))

    def decode_from_field(self, field: VisualFieldBundle, *, mirror_map: dict | None = None, diagnostic_modes: tuple[str, ...] = ()) -> dict[str, Any]:
        decode_started = time.perf_counter()
        first = self.category_decoder.first_pass(self.category_decoder.action_queries(), self.category_decoder.reason_queries(), field.action_context, field.reason_context)
        evidence_started = time.perf_counter()
        evidence = self.evidence_fields(field.evidence_layers, latent_layers=field.reason_layers)
        evidence_seconds = time.perf_counter() - evidence_started
        reread = self.rereader(first["action_tokens_direct"], first["reason_tokens_direct"], evidence, field.action_layers, field.reason_layers, first["action_logits_direct"], first["reason_logits_direct"])
        action_reread_tokens = first["action_tokens_direct"] + reread["action_reread_delta"]
        reason_reread_tokens = first["reason_tokens_direct"] + reread["reason_reread_delta"]
        exchange_certified = self.exchange(action_reread_tokens, reason_reread_tokens, evidence["explicit_tokens"], evidence["reliability"], "certified")
        exchange_ungated = self.exchange(action_reread_tokens, reason_reread_tokens, evidence["explicit_tokens"], evidence["reliability"], "ungated")
        action_final_tokens = action_reread_tokens + exchange_certified["action_exchange_delta"]
        latent_scores = torch.einsum("brd,bld->brl", self.reason_latent_query(reason_reread_tokens), evidence["latent_tokens"]) / (reason_reread_tokens.shape[-1] ** 0.5)
        latent_attention = torch.softmax(latent_scores, dim=-1)
        latent_message = torch.einsum("brl,bld->brd", latent_attention, self.reason_latent_value(evidence["latent_tokens"]))
        reason_latent_delta = 0.20 * torch.sigmoid(self.reason_latent_gamma_raw) * latent_message
        reason_semantic_tokens = reason_reread_tokens + exchange_certified["reason_exchange_delta"] + reason_latent_delta
        action_reread_logits = self.action_refined_head(action_reread_tokens).squeeze(-1)
        action_final_raw = self.action_refined_head(action_final_tokens).squeeze(-1)
        reason_semantic = self.reason_refined_head(reason_semantic_tokens).squeeze(-1)
        context = field.reason_context.mean(dim=1)
        annotation = self.annotation_head(reason_semantic_tokens, context, reason_semantic)
        threshold = self.threshold_head(action_final_raw, annotation["reason_logits_observed"])
        off = self.exchange(action_reread_tokens, reason_reread_tokens, evidence["explicit_tokens"], evidence["reliability"], "off")
        off_action = self.action_refined_head(action_reread_tokens + off["action_exchange_delta"]).squeeze(-1)
        off_reason = self.reason_refined_head(reason_reread_tokens + reason_latent_delta).squeeze(-1)
        explicit_reason = self.reason_refined_head(reason_reread_tokens + exchange_certified["reason_exchange_delta"]).squeeze(-1)
        latent_reason = self.reason_refined_head(reason_reread_tokens + reason_latent_delta).squeeze(-1)
        evidence_shuffled = self.exchange(action_reread_tokens, reason_reread_tokens, evidence["explicit_tokens"], evidence["reliability"], "evidence_shuffled")
        reason_shuffled = self.exchange(action_reread_tokens, reason_reread_tokens, evidence["explicit_tokens"], evidence["reliability"], "reason_tokens_shuffled")
        evidence_shuffled_reason = self.reason_refined_head(reason_reread_tokens + evidence_shuffled["reason_exchange_delta"] + reason_latent_delta).squeeze(-1)
        reason_shuffled_reason = self.reason_refined_head(reason_reread_tokens + reason_shuffled["reason_exchange_delta"] + reason_latent_delta).squeeze(-1)
        field_attention = evidence["field_attention"].clamp_min(1e-8)
        evidence_attention_entropy = -(field_attention * field_attention.log()).sum(-1).mean()
        evidence_effective_support = (field_attention > (1.0 / field_attention.shape[-1])).float().sum(-1).mean()
        output: dict[str, Any] = {
            **first,
            "action_logits_reread": action_reread_logits,
            "action_logits_exchange_ungated": self.action_refined_head(action_reread_tokens + exchange_ungated["action_exchange_delta"]).squeeze(-1),
            "action_logits_exchange_certified": action_final_raw,
            "action_logits_final_raw": action_final_raw,
            "reason_logits_semantic": reason_semantic,
            "reason_logits_final_raw": annotation["reason_logits_observed"],
            "action_tokens_final": action_final_tokens,
            "action_tokens_direct": first["action_tokens_direct"],
            "reason_tokens_direct": first["reason_tokens_direct"],
            "action_field_layers": field.action_layers,
            "reason_field_layers": field.reason_layers,
            "action_tokens_reread": action_reread_tokens,
            "reason_tokens_semantic": reason_semantic_tokens,
            "reason_latent_delta": reason_latent_delta,
            "reason_latent_attention": latent_attention,
            "reason_tokens_reread": reason_reread_tokens,
            "explicit_evidence_tokens": evidence["explicit_tokens"],
            "latent_evidence_tokens": evidence["latent_tokens"],
            "derived_atom_probs": evidence["derived_atom_probs"],
            "evidence_certificate_probability": evidence["certificate_probability"],
            "evidence_actor_part_type_logits": evidence["actor_part_type_logits"],
            "evidence_actor_part_occupancy_logits": evidence["actor_part_occupancy_logits"],
            "evidence_reliability": evidence["reliability"],
            "evidence_view_consistency": self.evidence_fields.view_consistency_ema.detach(),
            "semantic_weight_floor": self.semantic_weight_floor,
            "action_evidence_family_mask": self.exchange.family_mask_action,
            "evidence_presence_logits": evidence["presence_logits"],
            "evidence_observability_logits": evidence["observability_logits"],
            "evidence_state_logits": evidence["state_logits"],
            "evidence_state_channel_valid": evidence["state_channel_valid"],
            "evidence_part_coordinates": evidence["part_coordinates"],
            "evidence_part_scales": evidence["part_scales"],
            "evidence_masks": evidence["soft_masks"],
            "evidence_prototype_margin": evidence["prototype_margin"],
            "evidence_part_valid": evidence["part_valid"],
            "evidence_geometry_type": evidence["geometry_type"],
            "evidence_source_tokens": evidence["source_tokens"],
            "evidence_field_attention": evidence["field_attention"],
            "evidence_attention_entropy": evidence_attention_entropy,
            "evidence_effective_support": evidence_effective_support,
            "reason_token_shuffle_delta": (reason_semantic - reason_shuffled_reason).abs().mean(),
            "evidence_shuffle_delta": (reason_semantic - evidence_shuffled_reason).abs().mean(),
            **exchange_certified,
            **reread,
            **annotation,
            "action_logits_deploy": threshold["action_logits_deploy"],
            "reason_logits_deploy": threshold["reason_logits_deploy"],
            "action_logits_calibrated": threshold["action_logits_calibrated"],
            "reason_logits_calibrated": threshold["reason_logits_calibrated"],
        }
        output["branch_logits"] = {
            "action_direct": first["action_logits_direct"], "action_reread_no_exchange": action_reread_logits,
            "action_ungated_exchange": output["action_logits_exchange_ungated"], "action_certified_exchange": action_final_raw,
            "action_final_raw": action_final_raw, "action_deploy": output["action_logits_deploy"],
            "reason_direct": first["reason_logits_direct"], "reason_semantic": reason_semantic,
            "reason_observed": annotation["reason_logits_observed"], "reason_deploy": output["reason_logits_deploy"],
            "action_explicit_only": action_final_raw,
            "reason_explicit_only": explicit_reason,
            "action_latent_only": action_reread_logits,
            "reason_latent_only": latent_reason,
            "action_exchange_off": off_action,
            "reason_exchange_off": off_reason,
            "action_evidence_shuffled": self.action_refined_head(action_reread_tokens + evidence_shuffled["action_exchange_delta"]).squeeze(-1),
            "reason_evidence_shuffled": evidence_shuffled_reason,
            "action_reason_token_shuffled": self.action_refined_head(action_reread_tokens + reason_shuffled["action_exchange_delta"]).squeeze(-1),
            "reason_reason_token_shuffled": reason_shuffled_reason,
            "action_annotation_off": action_final_raw,
            "reason_annotation_off": reason_semantic,
        }
        output["diagnostics"] = {"dino_call_count": self.dino.dino_call_count, "explicit_only": True, "mirror_map_present": mirror_map is not None, "diagnostic_modes": diagnostic_modes, "evidence_seconds": evidence_seconds, "decode_seconds": time.perf_counter() - decode_started}
        return output

    def decode_cached_exchange(
        self,
        action_reread_tokens: torch.Tensor,
        reason_reread_tokens: torch.Tensor,
        explicit_evidence: torch.Tensor,
        reliability: torch.Tensor,
        reason_latent_delta: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Re-run only exchange and task heads for packed interventions."""
        with torch.autocast(device_type=action_reread_tokens.device.type, dtype=torch.bfloat16, enabled=action_reread_tokens.is_cuda):
            exchange = self.exchange(action_reread_tokens, reason_reread_tokens, explicit_evidence, reliability, "certified", evidence_grad=True)
            action = self.action_refined_head(action_reread_tokens + exchange["action_exchange_delta"]).squeeze(-1)
            latent = torch.zeros_like(reason_reread_tokens) if reason_latent_delta is None else reason_latent_delta
            reason = self.reason_refined_head(reason_reread_tokens + exchange["reason_exchange_delta"] + latent).squeeze(-1)
        return {"action_logits": action, "reason_logits": reason, **exchange}

    def decode_cached_intervention(
        self,
        action_tokens_direct: torch.Tensor,
        reason_tokens_direct: torch.Tensor,
        action_layers: torch.Tensor,
        reason_layers: torch.Tensor,
        action_logits_direct: torch.Tensor,
        reason_logits_direct: torch.Tensor,
        explicit_evidence: torch.Tensor,
        reliability: torch.Tensor,
        part_coordinates: torch.Tensor,
        part_valid: torch.Tensor,
        reason_latent_delta: torch.Tensor,
        field_enabled: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Recompute reread and exchange from cached DINO fields after evidence intervention."""
        with torch.autocast(device_type=action_tokens_direct.device.type, dtype=torch.bfloat16, enabled=action_tokens_direct.is_cuda):
            reread = self.rereader(
                action_tokens_direct, reason_tokens_direct,
                {"explicit_tokens": explicit_evidence, "part_coordinates": part_coordinates, "part_valid": part_valid, "reliability": reliability, "field_enabled": field_enabled},
                action_layers, reason_layers, action_logits_direct, reason_logits_direct,
            )
            action_reread = action_tokens_direct + reread["action_reread_delta"]
            reason_reread = reason_tokens_direct + reread["reason_reread_delta"]
        return self.decode_cached_exchange(action_reread, reason_reread, explicit_evidence, reliability, reason_latent_delta)

    def forward(self, images: torch.Tensor, *, mirror_map: dict | None = None, diagnostic_modes: tuple[str, ...] = ()) -> dict[str, Any]:
        # Evaluator and audit callers invoke model(images) directly, so they
        # need the same mixed-precision contract as the explicit train path.
        with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16, enabled=images.is_cuda):
            return self.decode_from_field(self.encode_images(images), mirror_map=mirror_map, diagnostic_modes=diagnostic_modes)

    def owned_parameters(self) -> dict[str, list[nn.Parameter]]:
        visual = self.visual_field.owned_parameters()
        return {
            "action_foundation": visual["action_foundation"],
            "action_decoder": [
                self.category_decoder.forward_query, self.category_decoder.stop_query,
                self.category_decoder.side_shared, self.category_decoder.left_embedding,
                self.category_decoder.right_embedding,
                *self.category_decoder.action_cross.parameters(),
                *self.category_decoder.action_self.parameters(),
                *self.category_decoder.action_head.parameters(),
                *self.action_refined_head.parameters(),
            ],
            "reason_semantic": visual["reason_semantic"] + list(self.category_decoder.reason_cross.parameters()) + list(self.category_decoder.reason_self.parameters()) + list(self.category_decoder.reason_head.parameters()) + list(self.reason_refined_head.parameters()) + list(self.category_decoder.entity.parameters()) + list(self.category_decoder.state.parameters()) + list(self.category_decoder.sector.parameters()) + list(self.category_decoder.role.parameters()) + list(self.category_decoder.reason_residual.parameters()) + self.evidence_fields.latent_parameters() + list(self.reason_latent_query.parameters()) + list(self.reason_latent_value.parameters()) + [self.reason_latent_gamma_raw],
            "evidence_core": visual["evidence_core"] + self.evidence_fields.explicit_parameters(),
            "exchange_reread": list(self.exchange.parameters()) + list(self.rereader.parameters()),
            "annotation_adapter": list(self.annotation_head.parameters()),
            "threshold_head": list(self.threshold_head.parameters()),
        }
