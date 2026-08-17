from __future__ import annotations

from typing import Any, Optional, Sequence, Union
import time

import torch
from torch import Tensor, nn

from .aie_calalign_foundation import AIECalAlignFoundation
from .aie_contribution_head import AIEContributionHead
from .aie_evidence_interface import AIEEvidenceInterface
from .aie_predicate_naming import AIEPredicateNaming
from .aie_reason_rereader import AIEReasonRereader


ActionScale = Union[float, Tensor, Sequence[float]]


class AIEOIAModel(nn.Module):
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
        self.action_dim = action_dim
        self.reason_dim = reason_dim
        self.foundation = AIECalAlignFoundation(
            dim=dim,
            action_dim=action_dim,
            reason_dim=reason_dim,
            selected_layers=selected_layers,
            pretrained_weights=pretrained_weights,
            scene_config=scene_config,
            grammar_path=grammar_path,
            use_mock_dino=use_mock_dino,
            mock_dim=mock_dim,
        )
        predicate_names = self.foundation.predicate_head.names
        self.action_evidence = AIEEvidenceInterface(
            dim=dim,
            action_dim=action_dim,
            probes_per_action=probes_per_action,
            num_layers=len(selected_layers),
            num_predicates=len(predicate_names),
            grid_hw=(45, 80),
            local_points_per_layer=local_points_per_layer,
            max_offset=max_offset,
            predicate_bias_max=predicate_bias_max,
            probe_chunk_size=probe_chunk_size,
        )
        self.action_contribution = AIEContributionHead(
            dim=dim,
            action_dim=action_dim,
            probes_per_action=probes_per_action,
            kappa=action_kappa,
            logit_norm_cap=action_logit_norm_cap,
        )
        self.predicate_naming = AIEPredicateNaming(dim=dim, num_predicates=len(predicate_names))
        self.reason_private = AIEReasonRereader(
            dim=dim,
            reason_dim=reason_dim,
            action_dim=action_dim,
            probes_per_action=probes_per_action,
            num_predicates=len(predicate_names),
            predicate_names=predicate_names,
            grammar_path=grammar_path,
            num_layers=len(selected_layers),
            kappa=reason_kappa,
        )

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        return self.foundation.encode_images(images)

    def decode_from_field(
        self,
        field: dict[str, Any],
        *,
        action_scale: ActionScale,
        reason_scale: float,
        predicate_bias_enabled: bool = True,
        local_reread_enabled: bool = True,
        group_attention_enabled: bool = True,
        action_evidence_shuffle: bool = False,
        wrong_action_evidence: bool = False,
        profile: bool = False,
        reason_action_prior_enabled: bool = True,
        reason_predicate_prior_enabled: bool = True,
        reason_action_scale: Optional[ActionScale] = None,
    ) -> dict[str, Any]:
        def stamp() -> float:
            sample = field["patch_tokens_by_layer"]
            if profile and sample.is_cuda:
                torch.cuda.synchronize(sample.device)
            return time.perf_counter()

        primary_start = stamp()
        primary = self.foundation.decode_field(field)
        primary_end = stamp()
        evidence = self.action_evidence(
            primary["action_nodes_primary"],
            primary["patch_tokens_by_layer_raw"],
            primary["predicate_attention"],
            primary["predicate_probs"],
            predicate_bias_enabled=predicate_bias_enabled,
            local_reread_enabled=local_reread_enabled,
            group_attention_enabled=group_attention_enabled,
            profile=profile,
        )
        evidence_end = stamp()
        if action_evidence_shuffle:
            # Deterministic batch rotation keeps the marginal distribution but breaks image ownership.
            permutation = torch.arange(evidence["evidence_token"].shape[0], device=evidence["evidence_token"].device).roll(1)
            for key in ("evidence_token", "evidence_map", "reference_point", "sampling_offsets", "sampling_weights"):
                evidence[key] = evidence[key][permutation]
        if wrong_action_evidence:
            # Rotate only the action axis; this diagnoses target-specific evidence use.
            for key in ("evidence_token", "evidence_map", "reference_point", "sampling_offsets", "sampling_weights"):
                evidence[key] = evidence[key].roll(1, dims=1)
        contribution = self.action_contribution(
            evidence["evidence_token"], primary["action_logits_primary"], action_scale=action_scale
        )
        reason_contribution = contribution
        if reason_action_scale is not None:
            reason_contribution = self.action_contribution(
                evidence["evidence_token"],
                primary["action_logits_primary"],
                action_scale=reason_action_scale,
            )
        naming = self.predicate_naming(
            evidence["evidence_token"],
            evidence["evidence_map"],
            primary["predicate_attention"],
            primary["predicate_probs"],
        )
        reason_start = stamp()
        reason = self.reason_private(
            primary["reason_nodes_primary"],
            primary["patch_tokens_by_layer_raw"],
            evidence["evidence_token"],
            evidence["evidence_map"],
            reason_contribution["bounded_contribution"],
            primary["predicate_attention"],
            primary["predicate_probs"],
            primary["reason_logits_primary"],
            reason_scale=reason_scale,
            action_prior_enabled=reason_action_prior_enabled,
            predicate_prior_enabled=reason_predicate_prior_enabled,
        )
        reason_end = stamp()
        result = {
            **primary,
            **evidence,
            **contribution,
            **naming,
            **reason,
            "reason_action_bounded_contribution": reason_contribution["bounded_contribution"],
            "evidence_reference_point": evidence["reference_point"],
            "evidence_sampling_offsets": evidence["sampling_offsets"],
            "evidence_layer_mixture": evidence["layer_mixture"],
            "branch_logits": {
                "primary_action": primary["action_logits_primary"],
                "final_action": contribution["action_logits_final"],
                "primary_reason": primary["reason_logits_primary"],
                "final_reason": reason["reason_logits_final"],
            },
        }
        if profile:
            result["_profile_primary_time"] = primary_end - primary_start
            result["_profile_evidence_global_time"] = evidence.get("_profile_evidence_global_time", evidence_end - primary_end)
            result["_profile_evidence_local_time"] = evidence.get("_profile_evidence_local_time", 0.0)
            result["_profile_reason_reread_time"] = reason_end - reason_start
        return result

    def rerun_action_evidence_from_field(
        self,
        modified_field: dict[str, Any],
        fixed_primary: dict[str, Tensor],
        *,
        action_scale: ActionScale,
        predicate_bias_enabled: bool,
    ) -> dict[str, Tensor]:
        evidence = self.action_evidence(
            fixed_primary["action_nodes_primary"].detach(),
            modified_field["patch_tokens_by_layer_raw"],
            fixed_primary["predicate_attention"].detach(),
            fixed_primary["predicate_probs"].detach(),
            predicate_bias_enabled=predicate_bias_enabled,
            local_reread_enabled=True,
        )
        contribution = self.action_contribution(
            evidence["evidence_token"], fixed_primary["action_logits_primary"].detach(), action_scale=action_scale
        )
        return {**evidence, **contribution}

    def forward(
        self,
        images: Tensor,
        *,
        action_scale: ActionScale = 1.0,
        reason_scale: float = 1.0,
    ) -> dict[str, Any]:
        return self.decode_from_field(
            self.encode_images(images),
            action_scale=action_scale,
            reason_scale=reason_scale,
        )
