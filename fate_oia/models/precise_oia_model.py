from __future__ import annotations

from pathlib import Path
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
from fate_oia.utils.precise_schema import load_evidence_fields, load_reason_semantics


class PRECISEOIAModel(nn.Module):
    def __init__(self, config_root: str | Path = "configs", pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth", use_mock_dino: bool = False) -> None:
        super().__init__()
        root = Path(config_root)
        self.reason_schema = load_reason_semantics(root / "precise_reason_semantics.yaml")
        self.evidence_schema = load_evidence_fields(root / "precise_evidence_fields.yaml")
        self.dino = PRECISEDinoFieldExtractor(pretrained_weights=pretrained_weights, use_mock_dino=use_mock_dino)
        self.visual_field = PRECISEVisualField()
        self.category_decoder = PRECISECategoryDecoder(self.reason_schema)
        self.evidence_fields = PRECISEEvidenceFields(self.evidence_schema)
        self.rereader = PRECISEVisualRereader()
        self.exchange = PRECISESemanticExchange(self.evidence_schema, self.reason_schema)
        self.annotation_head = PRECISEAnnotationHead()
        self.threshold_head = ACPRThresholdHead(action_dim=4, reason_dim=21)
        self.action_refined_head = nn.Linear(384, 1)
        self.reason_refined_head = nn.Linear(384, 1)

    def encode_images(self, images: torch.Tensor) -> VisualFieldBundle:
        return self.visual_field(self.dino(images))

    def decode_from_field(self, field: VisualFieldBundle, *, mirror_map: dict | None = None, diagnostic_modes: tuple[str, ...] = ()) -> dict[str, Any]:
        first = self.category_decoder.first_pass(self.category_decoder.action_queries(), self.category_decoder.reason_queries(), field.action_context, field.reason_context)
        evidence = self.evidence_fields(field.evidence_layers)
        reread = self.rereader(first["action_tokens_direct"], first["reason_tokens_direct"], evidence, field.action_layers, field.reason_layers)
        action_reread_tokens = first["action_tokens_direct"] + reread["action_reread_delta"]
        reason_reread_tokens = first["reason_tokens_direct"] + reread["reason_reread_delta"]
        exchange_certified = self.exchange(action_reread_tokens, reason_reread_tokens, evidence["explicit_tokens"], evidence["reliability"], "certified")
        exchange_ungated = self.exchange(action_reread_tokens, reason_reread_tokens, evidence["explicit_tokens"], evidence["reliability"], "ungated")
        action_final_tokens = action_reread_tokens + exchange_certified["action_exchange_delta"]
        reason_semantic_tokens = reason_reread_tokens + exchange_certified["reason_exchange_delta"]
        action_reread_logits = self.action_refined_head(action_reread_tokens).squeeze(-1)
        action_final_raw = self.action_refined_head(action_final_tokens).squeeze(-1)
        reason_semantic = self.reason_refined_head(reason_semantic_tokens).squeeze(-1)
        context = field.reason_context.mean(dim=1)
        annotation = self.annotation_head(reason_semantic_tokens, context, reason_semantic)
        threshold = self.threshold_head(action_final_raw, annotation["reason_logits_observed"])
        off = self.exchange(action_reread_tokens, reason_reread_tokens, evidence["explicit_tokens"], evidence["reliability"], "off")
        off_action = self.action_refined_head(action_reread_tokens + off["action_exchange_delta"]).squeeze(-1)
        output: dict[str, Any] = {
            **first,
            "action_logits_reread": action_reread_logits,
            "action_logits_exchange_ungated": self.action_refined_head(action_reread_tokens + exchange_ungated["action_exchange_delta"]).squeeze(-1),
            "action_logits_exchange_certified": action_final_raw,
            "action_logits_final_raw": action_final_raw,
            "reason_logits_semantic": reason_semantic,
            "reason_logits_final_raw": annotation["reason_logits_observed"],
            "action_tokens_final": action_final_tokens,
            "reason_tokens_semantic": reason_semantic_tokens,
            "explicit_evidence_tokens": evidence["explicit_tokens"],
            "latent_evidence_tokens": evidence["latent_tokens"],
            "derived_atom_probs": evidence["derived_atom_probs"],
            "evidence_reliability": evidence["reliability"],
            "evidence_presence_logits": evidence["presence_logits"],
            "evidence_observability_logits": evidence["observability_logits"],
            "evidence_state_logits": evidence["state_logits"],
            "evidence_part_coordinates": evidence["part_coordinates"],
            "evidence_masks": evidence["soft_masks"],
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
            "exchange_off": off_action,
        }
        output["diagnostics"] = {"dino_call_count": self.dino.dino_call_count, "explicit_only": True, "mirror_map_present": mirror_map is not None, "diagnostic_modes": diagnostic_modes}
        return output

    def forward(self, images: torch.Tensor, *, mirror_map: dict | None = None, diagnostic_modes: tuple[str, ...] = ()) -> dict[str, Any]:
        return self.decode_from_field(self.encode_images(images), mirror_map=mirror_map, diagnostic_modes=diagnostic_modes)

    def owned_parameters(self) -> dict[str, list[nn.Parameter]]:
        visual = self.visual_field.owned_parameters()
        return {
            "action_foundation": visual["action_foundation"],
            "action_decoder": list(self.category_decoder.action_cross.parameters()) + list(self.category_decoder.action_self.parameters()) + list(self.category_decoder.action_head.parameters()) + list(self.action_refined_head.parameters()),
            "reason_semantic": visual["reason_semantic"] + list(self.category_decoder.reason_cross.parameters()) + list(self.category_decoder.reason_self.parameters()) + list(self.category_decoder.reason_head.parameters()) + list(self.reason_refined_head.parameters()) + list(self.category_decoder.entity.parameters()) + list(self.category_decoder.state.parameters()) + list(self.category_decoder.sector.parameters()) + list(self.category_decoder.role.parameters()) + list(self.category_decoder.reason_residual.parameters()),
            "evidence_core": visual["evidence_core"] + list(self.evidence_fields.parameters()),
            "exchange_reread": list(self.exchange.parameters()) + list(self.rereader.parameters()),
            "annotation_adapter": list(self.annotation_head.parameters()),
            "threshold_head": list(self.threshold_head.parameters()),
        }
