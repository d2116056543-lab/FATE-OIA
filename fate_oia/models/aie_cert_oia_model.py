from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from .aie_cert_calalign_foundation import AIECertCalAlignFoundation
from .aie_cert_contribution_head import AIECertContributionHead
from .aie_cert_evidence_interface import AIECertEvidenceInterface
from .aie_cert_naming import AIECertNaming
from .aie_cert_reason_rereader import AIECertReasonRereader


class AIECertOIAModel(nn.Module):
    def __init__(self, dim=384, action_dim=4, reason_dim=21, selected_layers=(3, 7, 11),
                 pretrained_weights="ckp/reference/dino_deitsmall8_pretrain.pth",
                 scene_config="configs/aie_scene_predicates.yaml",
                 grammar_path="configs/acpr_reason_predicate_grammar.yaml", use_mock_dino=False, mock_dim=None):
        super().__init__()
        self.foundation = AIECertCalAlignFoundation(dim, action_dim, reason_dim, selected_layers,
            pretrained_weights, scene_config, grammar_path, use_mock_dino, mock_dim)
        names = self.foundation.predicate_head.names
        self.evidence_interface = AIECertEvidenceInterface(dim, action_dim, 4, len(selected_layers), len(names))
        self.contribution_head = AIECertContributionHead(dim, action_dim)
        self.reason_rereader = AIECertReasonRereader(dim, reason_dim, action_dim, len(names), names, grammar_path,
                                                     len(selected_layers))
        self.naming = AIECertNaming(dim)

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        return self.foundation.encode_images(images)

    def decode_from_field(self, field: dict[str, Any], *, action_scale=1.0, reason_budget_max=0.60,
                          predicate_prior_scale=1.0, transport_gamma_cap=0.25,
                          local_reread_enabled=True, transport_enabled=True, background_center_enabled=True,
                          action_residual_enabled=True, reason_action_prior_enabled=True,
                          reason_predicate_prior_enabled=True, reason_signed_priors=True,
                          reason_budget_enabled=True, reason_delta_enabled=True) -> dict[str, Any]:
        primary = self.foundation.decode_field(field)
        evidence = self.evidence_interface(primary["action_nodes_primary"], primary["patch_tokens_by_layer_raw"],
            primary["predicate_attention_clean"], primary["predicate_probs_clean"], primary["ego_region_masks"],
            prior_scale=predicate_prior_scale, gamma_cap=transport_gamma_cap,
            local_reread_enabled=local_reread_enabled, transport_enabled=transport_enabled,
            background_center_enabled=background_center_enabled)
        contribution = self.contribution_head(evidence["centered_atom_token"], primary["action_logits_primary"], action_scale)
        if not action_residual_enabled:
            contribution["action_delta"] = contribution["action_delta"].new_zeros(contribution["action_delta"].shape)
            contribution["bounded_contribution"] = contribution["bounded_contribution"].new_zeros(contribution["bounded_contribution"].shape)
            contribution["action_logits_final"] = primary["action_logits_primary"]
            contribution["action_logits_final_train"] = primary["action_logits_primary"].detach()
        reason = self.reason_rereader(primary["reason_nodes_primary"], primary["patch_tokens_by_layer_raw"],
            evidence["atom_token"], evidence["atom_map"], contribution["bounded_contribution"],
            primary["predicate_attention_clean"], primary["predicate_probs_clean"], primary["reason_logits_primary"],
            budget_max=reason_budget_max, action_prior_enabled=reason_action_prior_enabled,
            predicate_prior_enabled=reason_predicate_prior_enabled, signed_priors=reason_signed_priors,
            budget_enabled=reason_budget_enabled, delta_enabled=reason_delta_enabled)
        naming = self.naming(evidence["atom_token"], evidence["atom_map"], evidence["shared_predicate_keys"],
                             primary["predicate_attention_clean"], primary["predicate_probs_clean"])
        return {**primary, **evidence, **contribution, **reason, **naming,
            "branch_logits": {"primary_action": primary["action_logits_primary"],
                              "final_action": contribution["action_logits_final"],
                              "primary_reason": primary["reason_logits_primary"],
                              "final_reason": reason["reason_logits_final"]}}

    def forward(self, images: Tensor, **kwargs) -> dict[str, Any]:
        return self.decode_from_field(self.encode_images(images), **kwargs)
