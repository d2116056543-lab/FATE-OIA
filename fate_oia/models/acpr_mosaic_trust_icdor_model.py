from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from .acpr_dino_field import ACPRDinoFieldExtractor
from .mosaic_group_threshold import MOSAICGroupThresholdHead
from .mosaic_icdor_action_decoder import MOSAICICDORActionDecoder
from .mosaic_icdor_dual_reason_decoder import (
    MOSAICICDORLatentReasonDecoder,
    MOSAICICDORObservedReasonMixer,
    MOSAICICDORVisualReasonDecoder,
)
from .mosaic_icdor_observation_head import MOSAICICDORObservationHead
from .mosaic_low_rank_rezero_adapter import MOSAICLowRankReZeroPyramidAdapter
from .mosaic_masked_target_rereader import MOSAICMaskedTargetRereader
from .mosaic_native_semantics import load_icdor_ontology
from .mosaic_observable_predicates import MOSAICObservablePredicateLayer
from .mosaic_target_sparse_router import MOSAICTargetSparseRouter, _TIER_TO_ID
from .mosaic_visual_pyramid import MOSAICVisualPyramid


_PYRAMID_KEYS = ("F_hi", "F_mid", "F_ctx")


class MOSAICTrustICDORModel(nn.Module):
    """Direct-image IC-DOR model with strict action/reason/factor ownership."""

    def __init__(
        self,
        *,
        config_root: str | Path,
        backbone_arch: str = "vit_small",
        backbone_patch_size: int = 8,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        checkpoint_key: str = "teacher",
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        use_mock_dino: bool = False,
        mock_dim: int = 384,
        decoder_layers: int = 2,
        self_attention_heads: int = 4,
        highres_topk: int = 256,
        midres_topk: int = 128,
        anchors_per_factor: int = 2,
        typed_attention_heads: int = 4,
        point_samples: int = 4,
        curve_samples: int = 16,
        region_samples: int = 12,
        adapter_rank: int = 48,
        adapter_dropout: float = 0.05,
        adapter_rezero_init: float = 0.0,
        adapter_rezero_max: float = 0.30,
        spatial_prior_scale_init: float = 0.05,
        spatial_prior_scale_max: float = 0.20,
        spatial_prior_dropout: float = 0.50,
        content_temperature_init: float = 0.07,
    ) -> None:
        super().__init__()
        self.ontology = load_icdor_ontology(config_root)
        self.dino = ACPRDinoFieldExtractor(
            arch=backbone_arch,
            patch_size=backbone_patch_size,
            selected_layers=selected_layers,
            checkpoint_key=checkpoint_key,
            pretrained_weights=pretrained_weights,
            freeze_backbone=True,
            use_mock_dino=use_mock_dino,
            mock_dim=mock_dim,
        )
        dim = self.dino.dim
        # Separate trainable pyramids make the gradient firewall structural,
        # not merely a detach convention over a shared visual representation.
        self.factor_visual_pyramid = MOSAICVisualPyramid(input_dim=dim, output_dim=dim)
        self.action_visual_pyramid = MOSAICVisualPyramid(input_dim=dim, output_dim=dim)
        self.reason_visual_pyramid = MOSAICVisualPyramid(input_dim=dim, output_dim=dim)
        self.factor_adapter = MOSAICLowRankReZeroPyramidAdapter(
            dim=dim, rank=adapter_rank, dropout=adapter_dropout, rezero_init=adapter_rezero_init, rezero_max=adapter_rezero_max
        )
        self.action_adapter = MOSAICLowRankReZeroPyramidAdapter(
            dim=dim, rank=adapter_rank, dropout=adapter_dropout, rezero_init=adapter_rezero_init, rezero_max=adapter_rezero_max
        )
        self.reason_adapter = MOSAICLowRankReZeroPyramidAdapter(
            dim=dim, rank=adapter_rank, dropout=adapter_dropout, rezero_init=adapter_rezero_init, rezero_max=adapter_rezero_max
        )
        self.factor_extractor = MOSAICObservablePredicateLayer(
            self.ontology["factors"],
            dim=dim,
            anchors_per_factor=anchors_per_factor,
            heads=typed_attention_heads,
            point_samples=point_samples,
            curve_samples=curve_samples,
            region_samples=region_samples,
            prior_scale_init=spatial_prior_scale_init,
            prior_scale_max=spatial_prior_scale_max,
            prior_dropout=spatial_prior_dropout,
            content_temperature_init=content_temperature_init,
        )
        self.action_visual_decoder = MOSAICICDORActionDecoder(
            dim=dim,
            decoder_layers=decoder_layers,
            self_attention_heads=self_attention_heads,
            highres_topk=highres_topk,
            midres_topk=midres_topk,
        )
        self.action_router = MOSAICTargetSparseRouter(self.ontology, dim=dim)
        self.action_rereader = MOSAICMaskedTargetRereader(dim=dim, topk=highres_topk)
        self.reason_visual_decoder = MOSAICICDORVisualReasonDecoder(
            dim=dim,
            decoder_layers=decoder_layers,
            self_attention_heads=self_attention_heads,
            highres_topk=highres_topk,
            midres_topk=midres_topk,
        )
        self.reason_latent_decoder = MOSAICICDORLatentReasonDecoder(
            self.ontology,
            dim=dim,
            decoder_layers=decoder_layers,
            self_attention_heads=self_attention_heads,
            highres_topk=highres_topk,
            midres_topk=midres_topk,
        )
        self.observation_model = MOSAICICDORObservationHead(self.ontology)
        self.reason_observed_mixer = MOSAICICDORObservedReasonMixer()
        self.threshold_head = MOSAICGroupThresholdHead()
        factor_count = len(self.ontology["factors"])
        self.register_buffer("factor_certificate_tier", torch.zeros(factor_count, dtype=torch.long), persistent=True)
        self.register_buffer("factor_certificate_reliability", torch.zeros(factor_count), persistent=True)
        self.register_buffer("reason_factor_route_enabled", torch.zeros(factor_count, dtype=torch.bool), persistent=True)
        self.register_buffer("certificate_sha256_bytes", torch.zeros(32, dtype=torch.uint8), persistent=True)

    @staticmethod
    def _features(pyramid: dict[str, torch.Tensor | tuple[int, int]]) -> dict[str, torch.Tensor]:
        return {key: pyramid[key] for key in _PYRAMID_KEYS}  # type: ignore[return-value]

    @staticmethod
    def _detached_pyramid(pyramid: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Keep latent-reason supervision out of the direct visual-reason lane."""
        return {key: value.detach() for key, value in pyramid.items()}

    @torch.no_grad()
    def set_factor_certificate_tiers(self, tiers: Sequence[str], *, certificate_sha256: str | None = None) -> None:
        if len(tiers) != self.factor_certificate_tier.numel() or any(tier not in _TIER_TO_ID for tier in tiers):
            raise ValueError("IC-DOR factor certificate tiers do not match the factor ontology")
        tier_ids = torch.tensor([_TIER_TO_ID[tier] for tier in tiers], device=self.factor_certificate_tier.device)
        self.factor_certificate_tier.copy_(tier_ids)
        reliability = torch.where(
            tier_ids == _TIER_TO_ID["certified"],
            torch.ones_like(tier_ids, dtype=torch.float32),
            torch.where(
                tier_ids == _TIER_TO_ID["reason_only"],
                torch.full_like(tier_ids, 0.5, dtype=torch.float32),
                torch.zeros_like(tier_ids, dtype=torch.float32),
            ),
        )
        self.factor_certificate_reliability.copy_(reliability)
        self.reason_factor_route_enabled.copy_(tier_ids >= _TIER_TO_ID["reason_only"])
        self.action_router.set_certificate_tiers(tiers)
        if certificate_sha256 is not None:
            if len(certificate_sha256) != 64:
                raise ValueError("IC-DOR certificate sha256 must be 64 hexadecimal characters")
            self.certificate_sha256_bytes.copy_(torch.tensor(list(bytes.fromhex(certificate_sha256)), device=self.certificate_sha256_bytes.device))

    @torch.no_grad()
    def load_factor_certificate(self, certificate: Mapping[str, Any]) -> None:
        """Load one immutable audit certificate and reject test-derived or incomplete state."""
        if certificate.get("source_split") != "train_audit":
            raise ValueError("IC-DOR only accepts factor certificates derived from train_audit")
        entries = certificate.get("entries")
        digest = certificate.get("sha256")
        if not isinstance(entries, Mapping) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("IC-DOR factor certificate requires entries and a 64-character sha256")
        factor_names = [str(factor["name"]) for factor in self.ontology["factors"]]
        if set(entries) != set(factor_names):
            raise ValueError("IC-DOR factor certificate entries do not match the model ontology")
        tiers: list[str] = []
        expected_reliability = {"certified": 1.0, "reason_only": 0.5, "abstained": 0.0}
        for name in factor_names:
            entry = entries[name]
            if not isinstance(entry, Mapping) or entry.get("tier") not in expected_reliability:
                raise ValueError(f"IC-DOR factor certificate entry is invalid for {name}")
            tier = str(entry["tier"])
            value = entry.get("reliability")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != expected_reliability[tier]:
                raise ValueError(f"IC-DOR factor certificate reliability is invalid for {name}")
            tiers.append(tier)
        self.set_factor_certificate_tiers(tiers, certificate_sha256=digest.upper())

    @torch.no_grad()
    def set_edge_admission(self, edge_admission_mask: torch.Tensor) -> None:
        self.action_router.set_edge_admission(edge_admission_mask)

    @torch.no_grad()
    def set_route_gate_cap(self, cap: float) -> None:
        self.action_rereader.set_gate_cap(cap)

    @property
    def certificate_sha256(self) -> str:
        return bytes(self.certificate_sha256_bytes.detach().cpu().tolist()).hex().upper()

    def forward(
        self,
        images: torch.Tensor,
        *,
        route_mode: str = "auto",
        latent_enabled: bool = False,
        reason_route_mode: str = "full",
        prior_mode: str = "full",
        factor_ablation_mode: str = "full",
        factor_intervention_keep_mask: torch.Tensor | None = None,
        precomputed_dino_field: Mapping[str, Any] | None = None,
        return_masks: bool = False,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        if factor_ablation_mode not in {"full", "content_only", "prior_only", "query_shuffled", "image_shuffled"}:
            raise ValueError("IC-DOR factor ablation mode is invalid")
        if route_mode == "auto":
            if bool(self.action_router.edge_admission_mask.any()):
                route_mode = "admitted"
            elif bool((self.factor_certificate_tier == _TIER_TO_ID["certified"]).any()):
                route_mode = "shadow"
            else:
                route_mode = "off"
        if factor_ablation_mode in {"content_only", "prior_only"}:
            prior_mode = factor_ablation_mode
        # Audit interventions may reuse this batch-local field. It is never
        # persisted, never reused across samples/batches, and contains no labels.
        field = self.dino(images) if precomputed_dino_field is None else precomputed_dino_field
        if not isinstance(field, Mapping) or "patch_tokens_by_layer" not in field or "grid_hw" not in field:
            raise ValueError("IC-DOR precomputed DINO field is invalid")
        field_tokens = field["patch_tokens_by_layer"]
        if not isinstance(field_tokens, torch.Tensor) or field_tokens.shape[0] != images.shape[0]:
            raise ValueError("IC-DOR precomputed DINO field does not match image batch")
        if field_tokens.device != images.device:
            raise ValueError("IC-DOR precomputed DINO field must remain on the image device")
        patch_tokens = field["patch_tokens_by_layer"]
        factor_pyramid = self.factor_adapter(self._features(self.factor_visual_pyramid(patch_tokens)))
        action_pyramid = self.action_adapter(self._features(self.action_visual_pyramid(patch_tokens)))
        reason_pyramid = self.reason_adapter(self._features(self.reason_visual_pyramid(patch_tokens)))
        factor_input = {
            key: value.roll(shifts=1, dims=0) if factor_ablation_mode == "image_shuffled" else value
            for key, value in factor_pyramid.items()
        }
        factor_output = self.factor_extractor(factor_input, prior_mode=prior_mode)
        if factor_ablation_mode == "query_shuffled":
            for key, value in tuple(factor_output.items()):
                if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[1] == self.factor_certificate_tier.numel():
                    factor_output[key] = value.roll(shifts=1, dims=1)
        if factor_intervention_keep_mask is None:
            intervention_keep = images.new_ones(images.shape[0], self.factor_certificate_tier.numel())
        else:
            if factor_intervention_keep_mask.shape != (images.shape[0], self.factor_certificate_tier.numel()):
                raise ValueError("IC-DOR factor intervention keep mask must be [B,F]")
            intervention_keep = factor_intervention_keep_mask.to(device=images.device, dtype=images.dtype)
            if not torch.isfinite(intervention_keep).all() or bool(((intervention_keep < 0) | (intervention_keep > 1)).any()):
                raise ValueError("IC-DOR factor intervention keep mask must be finite in [0,1]")
        keep_feature = intervention_keep.unsqueeze(-1)
        keep_mask = intervention_keep.unsqueeze(-1).unsqueeze(-1)
        for key in ("factor_features",):
            factor_output[key] = factor_output[key] * keep_feature
        for key in ("factor_soft_masks",):
            factor_output[key] = factor_output[key] * keep_mask
        for key in (
            "factor_presence_prob", "factor_visibility_prob", "factor_positive_evidence", "factor_negative_evidence"
        ):
            factor_output[key] = factor_output[key] * intervention_keep
        factor_output["factor_uncertainty"] = (
            factor_output["factor_uncertainty"] * intervention_keep + (1.0 - intervention_keep)
        )
        for key in ("factor_presence_logits", "factor_visibility_logits"):
            factor_output[key] = torch.where(
                intervention_keep > 0.5,
                factor_output[key],
                torch.full_like(factor_output[key], -20.0),
            )
        action_output = self.action_visual_decoder(action_pyramid)
        router_output = self.action_router(
            factor_output["factor_features"],
            factor_output["factor_positive_evidence"],
            factor_output["factor_negative_evidence"],
            action_output["action_queries"],
            route_mode=route_mode,
        )
        reread_output = self.action_rereader(
            action_pyramid,
            action_output["action_queries"],
            factor_output["factor_soft_masks"],
            router_output["support_weights"],
            router_output["veto_weights"],
        )
        equal_mass_random_masks = torch.roll(
            factor_output["factor_soft_masks"],
            shifts=(factor_output["factor_soft_masks"].shape[-2] // 3, factor_output["factor_soft_masks"].shape[-1] // 3),
            dims=(-2, -1),
        )
        random_reread_output = None if route_mode == "off" else self.action_rereader(
            action_pyramid,
            action_output["action_queries"],
            equal_mass_random_masks,
            router_output["support_weights"],
            router_output["veto_weights"],
        )
        action_shadow = (
            action_output["action_visual_logits"]
            + reread_output["action_support_logits"]
            - reread_output["action_veto_logits"]
        )
        action_matched_random = action_output["action_visual_logits"] if random_reread_output is None else (
            action_output["action_visual_logits"]
            + random_reread_output["action_support_logits"]
            - random_reread_output["action_veto_logits"]
        )
        action_final = action_shadow if route_mode == "admitted" else action_output["action_visual_logits"]
        reason_visual = self.reason_visual_decoder(reason_pyramid)
        if reason_route_mode == "full":
            reason_reliability = self.factor_certificate_reliability
            reason_factor_features = factor_output["factor_features"] * reason_reliability.view(1, -1, 1)
            reason_factor_masks = factor_output["factor_soft_masks"] * reason_reliability.view(1, -1, 1, 1)
            reason_route_enabled = self.reason_factor_route_enabled
        elif reason_route_mode == "off":
            reason_reliability = self.factor_certificate_reliability
            reason_factor_features = factor_output["factor_features"] * reason_reliability.view(1, -1, 1)
            reason_factor_masks = factor_output["factor_soft_masks"] * reason_reliability.view(1, -1, 1, 1)
            reason_route_enabled = torch.zeros_like(self.reason_factor_route_enabled)
        elif reason_route_mode == "shuffled":
            # Preserve each target's learned route while permuting factor identity.
            # This is a real semantic-route ablation, not a repeated full metric.
            reason_reliability = self.factor_certificate_reliability.roll(shifts=1, dims=0)
            reason_factor_features = factor_output["factor_features"].roll(shifts=1, dims=1) * reason_reliability.view(1, -1, 1)
            reason_factor_masks = factor_output["factor_soft_masks"].roll(shifts=1, dims=1) * reason_reliability.view(1, -1, 1, 1)
            reason_route_enabled = self.reason_factor_route_enabled
        else:
            raise ValueError("IC-DOR reason_route_mode must be full, off, or shuffled")
        reason_latent = self.reason_latent_decoder(
            self._detached_pyramid(reason_pyramid),
            reason_factor_features,
            reason_factor_masks,
            reason_route_enabled,
        )
        observation = self.observation_model(
            reason_latent["reason_logits_latent"],
            factor_output["factor_visibility_prob"],
            factor_output["factor_uncertainty"],
        )
        reason_observed = self.reason_observed_mixer(
            reason_visual["reason_visual_observed_logits"],
            observation["reason_observation_logits"],
            latent_enabled=latent_enabled,
        )
        threshold = self.threshold_head(action_final, reason_observed["reason_observed_logits"])
        output: dict[str, torch.Tensor | dict[str, Any]] = {
            **factor_output,
            **action_output,
            **router_output,
            **reread_output,
            **reason_visual,
            **reason_latent,
            **observation,
            **reason_observed,
            **threshold,
            "action_shadow_logits": action_shadow,
            "action_matched_random_logits": action_matched_random,
            "equal_mass_random_factor_masks": equal_mass_random_masks,
            "action_final_logits": action_final,
            "action_logits_raw": action_final,
            "reason_logits_raw": reason_observed["reason_observed_logits"],
            "factor_certificate_tier": self.factor_certificate_tier,
            "factor_certificate_reliability": self.factor_certificate_reliability,
            "reason_factor_certificate_reliability_effective": reason_reliability,
            "reason_factor_route_enabled": self.reason_factor_route_enabled,
            "reason_factor_route_enabled_effective": reason_route_enabled,
            "dino_grid_hw": torch.tensor(field["grid_hw"], device=images.device),
            "factor_ablation_mode_code": images.new_tensor({
                "full": 0, "content_only": 1, "prior_only": 2, "query_shuffled": 3, "image_shuffled": 4
            }[factor_ablation_mode]),
            "factor_intervention_keep_mask": intervention_keep,
        }
        if return_diagnostics:
            shuffled_router = self.action_router(
                factor_output["factor_features"].roll(1, 1),
                factor_output["factor_positive_evidence"].roll(1, 1),
                factor_output["factor_negative_evidence"].roll(1, 1),
                action_output["action_queries"], route_mode=route_mode,
            )
            shuffled_read = self.action_rereader(
                action_pyramid, action_output["action_queries"],
                factor_output["factor_soft_masks"].roll(1, 1),
                shuffled_router["support_weights"], shuffled_router["veto_weights"],
            )
            wrong_target_read = self.action_rereader(
                action_pyramid, action_output["action_queries"], factor_output["factor_soft_masks"],
                router_output["support_weights"].roll(1, 2), router_output["veto_weights"].roll(1, 2),
            )
            output["action_factor_off_logits"] = action_output["action_visual_logits"]
            output["action_factor_shuffled_logits"] = (
                action_output["action_visual_logits"]
                + shuffled_read["action_support_logits"] - shuffled_read["action_veto_logits"]
            )
            output["action_wrong_target_logits"] = (
                action_output["action_visual_logits"]
                + wrong_target_read["action_support_logits"] - wrong_target_read["action_veto_logits"]
            )
            output["action_equal_mass_random_logits"] = action_matched_random
            for diagnostic_mode in ("off", "shuffled"):
                if diagnostic_mode == "off":
                    diagnostic_features = factor_output["factor_features"] * self.factor_certificate_reliability.view(1, -1, 1)
                    diagnostic_masks = factor_output["factor_soft_masks"] * self.factor_certificate_reliability.view(1, -1, 1, 1)
                    diagnostic_enabled = torch.zeros_like(self.reason_factor_route_enabled)
                else:
                    reliability = self.factor_certificate_reliability.roll(1, 0)
                    diagnostic_features = factor_output["factor_features"].roll(1, 1) * reliability.view(1, -1, 1)
                    diagnostic_masks = factor_output["factor_soft_masks"].roll(1, 1) * reliability.view(1, -1, 1, 1)
                    diagnostic_enabled = self.reason_factor_route_enabled
                diagnostic_latent = self.reason_latent_decoder(
                    self._detached_pyramid(reason_pyramid), diagnostic_features, diagnostic_masks, diagnostic_enabled
                )
                diagnostic_observation = self.observation_model(
                    diagnostic_latent["reason_logits_latent"], factor_output["factor_visibility_prob"], factor_output["factor_uncertainty"]
                )
                diagnostic_observed = self.reason_observed_mixer(
                    reason_visual["reason_visual_observed_logits"], diagnostic_observation["reason_observation_logits"],
                    latent_enabled=latent_enabled,
                )
                output[f"reason_observed_logits_route_{diagnostic_mode}"] = diagnostic_observed["reason_observed_logits"]
        if not return_masks:
            for key in (
                "factor_soft_masks",
                "equal_mass_random_factor_masks",
                "sampling_coordinates",
                "action_support_mask",
                "action_veto_mask",
                "action_support_attention",
                "action_veto_attention",
                "reason_factor_masks",
            ):
                output.pop(key, None)
        return output
