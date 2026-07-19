from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .mosaic_group_threshold import MOSAICGroupThresholdHead
from .mosaic_action_route_policy import compose_final_action_logits
from .mosaic_batch_field_reuse import BatchLocalDinoFieldReuse
from .mosaic_continuous_credibility import ContinuousVisualCredibility
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
from .mosaic_target_utility import MOSAICAuditTargetUtility
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
        gate_init: float = 0.02,
        gate_max: float = 0.15,
        action_shadow_credibility_floor: float = 0.05,
        reason_semantic_credibility_floor: float = 0.15,
        router_dustbin_init: float = -4.0,
        pi_min: float = 0.20,
        pi_max: float = 0.95,
        observed_mix_init: float = 0.05,
        credibility_independent_of_reason_labels: bool = True,
        credibility_ema_decay: float = 0.90,
        credibility_image_only_cap: float = 0.10,
        credibility_unknown_cap: float = 0.0,
        credibility_no_reliable_negative_cap: float = 0.25,
        fine_transport_enabled: bool = True,
        fine_eta_by_type: Mapping[str, float] | None = None,
        local_reread_offset_max: float = 0.08,
        fine_off_diagnostic: bool = True,
        coarse_off_diagnostic: bool = True,
    ) -> None:
        super().__init__()
        if credibility_independent_of_reason_labels is not True:
            raise ValueError("CREDO visual credibility must remain independent of reason labels")
        if not 0.0 <= float(action_shadow_credibility_floor) <= 1.0:
            raise ValueError("CREDO action shadow training-access floor must be in [0, 1]")
        if not 0.0 <= float(reason_semantic_credibility_floor) <= 1.0:
            raise ValueError("CREDO reason semantic training-access floor must be in [0, 1]")
        self.credibility_independent_of_reason_labels = True
        self.action_shadow_credibility_floor = float(action_shadow_credibility_floor)
        self.reason_semantic_credibility_floor = float(reason_semantic_credibility_floor)
        self.fine_transport_diagnostics = {
            "fine_off": bool(fine_off_diagnostic),
            "coarse_off": bool(coarse_off_diagnostic),
        }
        self.ontology = load_icdor_ontology(config_root)
        # Keep public router/decoder imports usable when the optional DINO
        # checkout is unavailable; construction still requires it as before.
        from .acpr_dino_field import ACPRDinoFieldExtractor

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
        # Reuse one DINO field for same-batch branch ablations only. The
        # trainer clears this explicitly at every batch boundary.
        self._batch_field_reuse = BatchLocalDinoFieldReuse(self.dino)
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
            fine_transport_enabled=fine_transport_enabled,
            fine_eta_by_type=fine_eta_by_type,
        )
        self.continuous_credibility = ContinuousVisualCredibility(
            factor_count=len(self.ontology["factors"]),
            dim=dim,
            factor_roles=tuple(str(factor.get("role", "observable")) for factor in self.ontology["factors"]),
            source_kinds=tuple(str(factor.get("source_kind", "grounded")) for factor in self.ontology["factors"]),
            ema_decay=credibility_ema_decay,
            image_only_cap=credibility_image_only_cap,
            unknown_cap=credibility_unknown_cap,
            no_reliable_negative_cap=credibility_no_reliable_negative_cap,
        )
        self.action_visual_decoder = MOSAICICDORActionDecoder(
            dim=dim,
            decoder_layers=decoder_layers,
            self_attention_heads=self_attention_heads,
            highres_topk=highres_topk,
            midres_topk=midres_topk,
        )
        self.action_router = MOSAICTargetSparseRouter(
            self.ontology, dim=dim, dustbin_init=router_dustbin_init
        )
        self.target_utility = MOSAICAuditTargetUtility(
            factor_count=len(self.ontology["factors"]), reason_count=21, action_count=4
        )
        self.action_rereader = MOSAICMaskedTargetRereader(
            dim=dim,
            topk=highres_topk,
            gate_init=gate_init,
            gate_max=gate_max,
            local_reread_offset_max=local_reread_offset_max,
        )
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
            local_reread_offset_max=local_reread_offset_max,
        )
        self.observation_model = MOSAICICDORObservationHead(self.ontology, pi_min=pi_min, pi_max=pi_max)
        self.reason_observed_mixer = MOSAICICDORObservedReasonMixer(init_mix=observed_mix_init)
        self.threshold_head = MOSAICGroupThresholdHead()
        factor_count = len(self.ontology["factors"])
        self.register_buffer("factor_certificate_tier", torch.zeros(factor_count, dtype=torch.long), persistent=True)
        self.register_buffer("factor_certificate_reliability", torch.zeros(factor_count), persistent=True)
        self.register_buffer("reason_factor_route_enabled", torch.zeros(factor_count, dtype=torch.bool), persistent=True)
        # PU recovery is admitted independently per reason. The latent core
        # remains trained even while every entry here is false.
        self.register_buffer("reason_pu_gate", torch.zeros(21, dtype=torch.bool), persistent=True)
        self.register_buffer("certificate_sha256_bytes", torch.zeros(32, dtype=torch.uint8), persistent=True)

    @torch.no_grad()
    def update_reason_pu_gate(self, gate: torch.Tensor) -> None:
        """Install the train-audit-derived per-label PU admission gate."""
        if gate.shape != (21,) or gate.dtype != torch.bool:
            raise ValueError("IC-DOR reason PU gate must be bool [21]")
        self.reason_pu_gate.copy_(gate.to(device=self.reason_pu_gate.device))

    @staticmethod
    def _features(pyramid: dict[str, torch.Tensor | tuple[int, int]]) -> dict[str, torch.Tensor]:
        return {key: pyramid[key] for key in _PYRAMID_KEYS}  # type: ignore[return-value]

    @staticmethod
    def _detached_pyramid(pyramid: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Keep latent-reason supervision out of the direct visual-reason lane."""
        return {key: value.detach() for key, value in pyramid.items()}

    @staticmethod
    def _typed_mask_values(
        factor_masks: torch.Tensor, sampling_coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Sample the active factor mask at every typed evidence coordinate."""
        if factor_masks.ndim != 4 or sampling_coordinates.ndim != 6:
            raise ValueError("IC-DOR typed mask sampling requires [B,F,H,W] and [B,F,A,H,S,2]")
        batch_size, factor_count, height, width = factor_masks.shape
        if sampling_coordinates.shape[:2] != (batch_size, factor_count):
            raise ValueError("IC-DOR typed coordinates do not match factor masks")
        sample_shape = sampling_coordinates.shape[2:-1]
        flat_count = int(torch.tensor(sample_shape).prod().item())
        grid = sampling_coordinates.reshape(batch_size * factor_count, flat_count, 1, 2)
        sampled = F.grid_sample(
            factor_masks.reshape(batch_size * factor_count, 1, height, width),
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return sampled.reshape(batch_size, factor_count, *sample_shape).clamp_min(0.0)

    @staticmethod
    def _resample_typed_features(
        feature_map: torch.Tensor, sampling_coordinates: torch.Tensor
    ) -> torch.Tensor:
        """Re-read a factor's typed samples after a spatial control changes it."""
        if feature_map.ndim != 4 or sampling_coordinates.ndim != 6:
            raise ValueError("IC-DOR typed feature sampling requires [B,D,H,W] and [B,F,A,H,S,2]")
        batch_size, dim, height, width = feature_map.shape
        if sampling_coordinates.shape[0] != batch_size:
            raise ValueError("IC-DOR typed coordinates do not match the feature map batch")
        factor_count = sampling_coordinates.shape[1]
        sample_shape = sampling_coordinates.shape[2:-1]
        flat_count = int(torch.tensor(sample_shape).prod().item())
        grid = sampling_coordinates.reshape(batch_size * factor_count, flat_count, 1, 2)
        repeated_map = feature_map.unsqueeze(1).expand(-1, factor_count, -1, -1, -1)
        sampled = F.grid_sample(
            repeated_map.reshape(batch_size * factor_count, dim, height, width),
            grid,
            mode="bilinear",
            align_corners=False,
        )
        return sampled.squeeze(-1).transpose(1, 2).reshape(
            batch_size, factor_count, *sample_shape, dim
        )

    @staticmethod
    def _rolled_typed_coordinates(
        sampling_coordinates: torch.Tensor, *, height: int, width: int
    ) -> torch.Tensor:
        """Match the same cyclic spatial roll used by the equal-mass control mask."""
        shift_y, shift_x = height // 3, width // 3
        offset = sampling_coordinates.new_tensor((2.0 * shift_x / width, 2.0 * shift_y / height))
        # ``torch.roll`` is cyclic, so the coordinate control must wrap rather
        # than clamp at an image boundary and accidentally change mask mass.
        return torch.remainder(sampling_coordinates + offset + 1.0, 2.0) - 1.0

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
        if certificate.get("source_split") != "audit_visual":
            raise ValueError("CREDO only accepts factor certificates derived from audit_visual")
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
    def clear_batch_field_reuse(self) -> None:
        """Prevent a field from surviving into the next input batch."""
        self._batch_field_reuse.clear()

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
        factor_mask_mode: str = "configured",
        factor_intervention_keep_mask: torch.Tensor | None = None,
        factor_mask_override: torch.Tensor | None = None,
        precomputed_dino_field: Mapping[str, Any] | None = None,
        return_masks: bool = False,
        return_diagnostics: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        if factor_ablation_mode not in {"full", "content_only", "prior_only", "query_shuffled", "image_shuffled"}:
            raise ValueError("IC-DOR factor ablation mode is invalid")
        if factor_mask_mode not in {"configured", "fine", "coarse"}:
            raise ValueError("IC-DOR factor mask mode must be configured, fine, or coarse")
        if route_mode == "auto":
            # Certificate state is deployment evidence only.  Learning access
            # is continuous and starts with a shadow route from epoch zero.
            route_mode = "admitted" if bool(self.action_router.edge_admission_mask.any()) else "shadow"
        if factor_ablation_mode in {"content_only", "prior_only"}:
            prior_mode = factor_ablation_mode
        # Audit interventions may reuse this batch-local field. It is never
        # persisted, never reused across samples/batches, and contains no labels.
        field = (
            self._batch_field_reuse(images)
            if precomputed_dino_field is None
            else precomputed_dino_field
        )
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
        query_permutation = (
            torch.roll(torch.arange(self.factor_certificate_tier.numel(), device=images.device), shifts=1)
            if factor_ablation_mode == "query_shuffled"
            else None
        )
        factor_output = self.factor_extractor(
            factor_input,
            prior_mode=prior_mode,
            query_permutation=query_permutation,
        )
        # The V5 prior-gap objective compares the visual measurement against
        # the extractor's own no-image branch while reusing this batch's DINO
        # field and pyramid. It never becomes an action/reason input.
        if return_diagnostics:
            prior_factor_output = self.factor_extractor(factor_pyramid, prior_mode="prior_only")
            factor_output["factor_prior_presence_logits"] = prior_factor_output["factor_presence_logits"]
        if factor_mask_mode == "fine":
            factor_output["factor_soft_masks"] = factor_output["factor_fine_masks"]
        elif factor_mask_mode == "coarse":
            factor_output["factor_soft_masks"] = factor_output["factor_coarse_masks"]
        sample_support = (factor_output["sample_attention"] > 1e-5).float().sum(dim=(-1, -2, -3))
        credibility = self.continuous_credibility(
            factor_output["factor_features"],
            factor_output["factor_presence_prob"],
            factor_output["factor_uncertainty"],
            grounding_score=(
                sample_support
                / float(max(1, factor_output["sample_attention"].shape[-3] * factor_output["sample_attention"].shape[-2] * factor_output["sample_attention"].shape[-1]))
            ).clamp(0.0, 1.0),
            sample_support=sample_support,
        )
        # Only the completed audit_visual pass may update or provide routing
        # credibility. Per-batch measurements remain diagnostics, never a
        # same-epoch self-certification signal.
        stored_credibility = self.continuous_credibility.ema_cV.to(
            device=images.device,
            dtype=factor_output["factor_features"].dtype,
        ).view(1, -1).expand(images.shape[0], -1)
        credibility["cV_batch_measurement"] = credibility["cV"]
        credibility["cV"] = stored_credibility
        credibility["cV_ema"] = stored_credibility
        factor_output.update(credibility)
        # Training access and deployment admission are deliberately distinct.
        # These floors are only a learning-access interpolation prescribed by
        # CREDO-MAP. They never enter the final-edge admission predicate, and
        # raw cV remains exported unchanged for audit/calibration.
        continuous_route_weight = stored_credibility.detach().clamp(0.0, 1.0)
        action_route_train_access = self.action_shadow_credibility_floor + (
            1.0 - self.action_shadow_credibility_floor
        ) * continuous_route_weight
        reason_route_train_access = self.reason_semantic_credibility_floor + (
            1.0 - self.reason_semantic_credibility_floor
        ) * continuous_route_weight
        factor_output["cV_route_effective"] = continuous_route_weight
        factor_output["cV_action_shadow_training_access"] = action_route_train_access
        factor_output["cV_reason_training_access"] = reason_route_train_access
        target_utility_state = self.target_utility()
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
        keep_typed_attention = intervention_keep.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        keep_typed_features = keep_typed_attention.unsqueeze(-1)
        for key in ("factor_features",):
            factor_output[key] = factor_output[key] * keep_feature
        for key in ("factor_soft_masks", "factor_fine_masks", "factor_coarse_masks"):
            factor_output[key] = factor_output[key] * keep_mask
        for key in (
            "factor_presence_prob", "factor_visibility_prob", "factor_positive_evidence", "factor_negative_evidence"
        ):
            factor_output[key] = factor_output[key] * intervention_keep
        # Deletion must clear the direct typed-sample path as well as the
        # coarse mask. Otherwise an intervened factor can still reach an
        # action/reason target through stale sampled features.
        factor_output["sample_attention"] = factor_output["sample_attention"] * keep_typed_attention
        factor_output["sampled_features"] = factor_output["sampled_features"] * keep_typed_features
        factor_output["factor_uncertainty"] = (
            factor_output["factor_uncertainty"] * intervention_keep + (1.0 - intervention_keep)
        )
        for key in ("factor_presence_logits", "factor_visibility_logits"):
            factor_output[key] = torch.where(
                intervention_keep > 0.5,
                factor_output[key],
                torch.full_like(factor_output[key], -20.0),
            )
        if factor_mask_override is not None:
            expected = factor_output["factor_soft_masks"].shape
            if factor_mask_override.shape != expected:
                raise ValueError("IC-DOR factor mask override must be [B,F,H,W]")
            if not torch.isfinite(factor_mask_override).all() or bool((factor_mask_override < 0).any()):
                raise ValueError("IC-DOR factor mask override must be finite and non-negative")
            # Audit collectors construct this from the current image-only batch;
            # it only changes spatial rereads and is never persisted as model state.
            factor_output["factor_soft_masks"] = factor_mask_override.to(
                device=images.device, dtype=factor_output["factor_soft_masks"].dtype
            )
        # A coarse mask override/deletion is not a real intervention unless it
        # also masks the typed path. The target rereaders otherwise retain
        # stale local samples even after a factor was removed from its mask.
        typed_mask_values = self._typed_mask_values(
            factor_output["factor_soft_masks"], factor_output["sampling_coordinates"]
        )
        factor_output["sample_attention"] = factor_output["sample_attention"] * typed_mask_values
        factor_output["sampled_features"] = factor_output["sampled_features"] * typed_mask_values.unsqueeze(-1)
        # Factor measurement has its own objective. Action shadow may read
        # its evidence, but must not update the factor visual lane.
        action_factor_features = factor_output["factor_features"].detach()
        action_positive_evidence = factor_output["factor_positive_evidence"].detach()
        action_negative_evidence = factor_output["factor_negative_evidence"].detach()
        action_factor_masks = factor_output["factor_soft_masks"].detach()
        action_sampling_coordinates = factor_output["sampling_coordinates"].detach()
        action_sampled_features = factor_output["sampled_features"].detach()
        action_sample_attention = factor_output["sample_attention"].detach()
        action_output = self.action_visual_decoder(action_pyramid)
        router_output = self.action_router(
            action_factor_features,
            action_positive_evidence,
            action_negative_evidence,
            action_output["action_queries"],
            route_mode=route_mode,
            factor_credibility=action_route_train_access,
            factor_target_utility=target_utility_state["action_target_utility"],
        )
        reread_output = self.action_rereader(
            action_pyramid,
            action_output["action_queries"],
            action_factor_masks,
            router_output["support_weights"],
            router_output["veto_weights"],
            action_sampling_coordinates,
            action_sampled_features,
            action_sample_attention,
        )
        equal_mass_random_masks = torch.roll(
            action_factor_masks,
            shifts=(action_factor_masks.shape[-2] // 3, action_factor_masks.shape[-1] // 3),
            dims=(-2, -1),
        )
        random_sampling_coordinates = self._rolled_typed_coordinates(
            action_sampling_coordinates,
            height=equal_mass_random_masks.shape[-2],
            width=equal_mass_random_masks.shape[-1],
        )
        random_sample_attention = self._typed_mask_values(
            equal_mass_random_masks, random_sampling_coordinates
        )
        random_sampled_features = self._resample_typed_features(
            factor_pyramid["F_hi"].detach(), random_sampling_coordinates
        ) * random_sample_attention.unsqueeze(-1)
        random_reread_output = None if route_mode == "off" else self.action_rereader(
            action_pyramid,
            action_output["action_queries"],
            equal_mass_random_masks,
            router_output["support_weights"],
            router_output["veto_weights"],
            random_sampling_coordinates,
            random_sampled_features,
            random_sample_attention,
        )
        action_shadow = (
            action_output["action_visual_logits"].detach()
            + reread_output["action_support_logits"]
            - reread_output["action_veto_logits"]
        )
        # The same-image random control is a shadow-route objective. Anchor
        # it to a stopped visual baseline so its loss cannot update the direct
        # visual action owner through the matched-control branch.
        action_matched_random = action_output["action_visual_logits"].detach() if random_reread_output is None else (
            action_output["action_visual_logits"].detach()
            + random_reread_output["action_support_logits"]
            - random_reread_output["action_veto_logits"]
        )
        admitted = (
            self.action_router.edge_admission_mask.any(dim=(0, 1))
            if route_mode == "admitted"
            else torch.zeros(self.action_router.action_count, dtype=torch.bool, device=images.device)
        )
        action_final = compose_final_action_logits(action_output["action_visual_logits"], action_shadow, admitted)
        reason_visual = self.reason_visual_decoder(reason_pyramid)
        if reason_route_mode == "full":
            reason_reliability = reason_route_train_access
            reason_factor_features = factor_output["factor_features"] * reason_reliability.unsqueeze(-1)
            reason_factor_masks = factor_output["factor_soft_masks"] * reason_reliability.unsqueeze(-1).unsqueeze(-1)
            reason_route_enabled = torch.ones_like(self.reason_factor_route_enabled)
            reason_positive_evidence = factor_output["factor_positive_evidence"] * reason_reliability
            reason_negative_evidence = factor_output["factor_negative_evidence"] * reason_reliability
            reason_sampling_coordinates = factor_output["sampling_coordinates"]
            reason_sampled_features = factor_output["sampled_features"] * reason_reliability.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            reason_sample_attention = factor_output["sample_attention"] * reason_reliability.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        elif reason_route_mode == "off":
            reason_reliability = torch.zeros_like(continuous_route_weight)
            reason_factor_features = factor_output["factor_features"] * 0.0
            reason_factor_masks = factor_output["factor_soft_masks"] * 0.0
            reason_route_enabled = torch.zeros_like(self.reason_factor_route_enabled)
            reason_positive_evidence = factor_output["factor_positive_evidence"] * 0.0
            reason_negative_evidence = factor_output["factor_negative_evidence"] * 0.0
            reason_sampling_coordinates = factor_output["sampling_coordinates"]
            reason_sampled_features = factor_output["sampled_features"] * 0.0
            reason_sample_attention = factor_output["sample_attention"] * 0.0
        elif reason_route_mode == "shuffled":
            # Preserve each target's learned route while permuting factor identity.
            # This is a real semantic-route ablation, not a repeated full metric.
            reason_reliability = continuous_route_weight.roll(shifts=1, dims=1)
            reason_factor_features = factor_output["factor_features"].roll(shifts=1, dims=1) * reason_reliability.unsqueeze(-1)
            reason_factor_masks = factor_output["factor_soft_masks"].roll(shifts=1, dims=1) * reason_reliability.unsqueeze(-1).unsqueeze(-1)
            reason_route_enabled = torch.ones_like(self.reason_factor_route_enabled)
            reason_positive_evidence = factor_output["factor_positive_evidence"].roll(shifts=1, dims=1)
            reason_negative_evidence = factor_output["factor_negative_evidence"].roll(shifts=1, dims=1)
            reason_sampling_coordinates = factor_output["sampling_coordinates"].roll(shifts=1, dims=1)
            reason_sampled_features = factor_output["sampled_features"].roll(shifts=1, dims=1)
            reason_sample_attention = factor_output["sample_attention"].roll(shifts=1, dims=1)
        else:
            raise ValueError("IC-DOR reason_route_mode must be full, off, or shuffled")
        reason_latent = self.reason_latent_decoder(
            self._detached_pyramid(reason_pyramid),
            reason_factor_features,
            reason_factor_masks,
            reason_route_enabled,
            reason_positive_evidence.detach(),
            reason_negative_evidence.detach(),
            reason_sampling_coordinates,
            reason_sampled_features,
            reason_sample_attention,
            target_utility_state["semantic_compatibility"],
        )
        observation = self.observation_model(
            reason_latent["reason_logits_latent"],
            # Observation/reason loss must not update the visual factor
            # extractor. Factor supervision owns those measurements.
            factor_output["factor_visibility_prob"].detach(),
            factor_output["factor_uncertainty"].detach(),
        )
        reason_observed = self.reason_observed_mixer(
            reason_visual["reason_visual_observed_logits"],
            observation["reason_observation_logits"],
            latent_enabled=latent_enabled,
            route_mass=reason_latent["reason_route_mass"],
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
            "action_matched_random_sampling_coordinates": random_sampling_coordinates,
            "action_typed_sample_attention_effective": factor_output["sample_attention"],
            "action_final_logits": action_final,
            "action_logits_raw": action_final,
            "action_visual_logits": action_output["action_visual_logits"],
            "reason_logits_raw": reason_observed["reason_observed_logits"],
            "reason_visual_logits": reason_visual["reason_visual_observed_logits"],
            "reason_latent_logits": reason_latent["reason_logits_latent"],
            "reason_final_logits": reason_observed["reason_observed_logits"],
            "factor_certificate_tier": self.factor_certificate_tier,
            "factor_certificate_reliability": self.factor_certificate_reliability,
            "reason_factor_certificate_reliability_effective": reason_reliability,
            "reason_factor_route_enabled": self.reason_factor_route_enabled,
            "reason_pu_gate": self.reason_pu_gate,
            "reason_factor_route_enabled_effective": reason_route_enabled,
            "reason_continuous_credibility": continuous_route_weight,
            "semantic_compatibility": target_utility_state["semantic_compatibility"],
            "action_target_utility": target_utility_state["action_target_utility"],
            "target_utility_initialized": target_utility_state["target_utility_initialized"],
            "dino_grid_hw": torch.tensor(field["grid_hw"], device=images.device),
            "factor_ablation_mode_code": images.new_tensor({
                "full": 0, "content_only": 1, "prior_only": 2, "query_shuffled": 3, "image_shuffled": 4
            }[factor_ablation_mode]),
            "factor_mask_mode_code": images.new_tensor({"configured": 0, "fine": 1, "coarse": 2}[factor_mask_mode]),
            "fine_transport_enabled": images.new_tensor(float(self.factor_extractor.fine_transport_enabled)),
            "factor_intervention_keep_mask": intervention_keep,
        }
        if random_reread_output is not None:
            output["action_matched_random_typed_target_coordinates"] = random_reread_output[
                "action_typed_target_coordinates"
            ]
        if return_diagnostics:
            shuffled_router = self.action_router(
                factor_output["factor_features"].roll(1, 1),
                factor_output["factor_positive_evidence"].roll(1, 1),
                factor_output["factor_negative_evidence"].roll(1, 1),
                action_output["action_queries"], route_mode=route_mode,
                # Keep the shuffled control on the same V5 learning-access
                # scale as the live action router. Raw cV is audit-only.
                factor_credibility=action_route_train_access.roll(1, 1),
                factor_target_utility=target_utility_state["action_target_utility"].roll(1, 0),
            )
            shuffled_read = self.action_rereader(
                action_pyramid, action_output["action_queries"],
                factor_output["factor_soft_masks"].roll(1, 1),
                shuffled_router["support_weights"], shuffled_router["veto_weights"],
                factor_output["sampling_coordinates"].roll(1, 1),
                factor_output["sampled_features"].roll(1, 1),
                factor_output["sample_attention"].roll(1, 1),
            )
            wrong_target_read = self.action_rereader(
                action_pyramid, action_output["action_queries"], factor_output["factor_soft_masks"],
                router_output["support_weights"].roll(1, 2), router_output["veto_weights"].roll(1, 2),
                factor_output["sampling_coordinates"], factor_output["sampled_features"],
                factor_output["sample_attention"],
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
                    diagnostic_features = factor_output["factor_features"] * 0.0
                    diagnostic_masks = factor_output["factor_soft_masks"] * 0.0
                    diagnostic_enabled = torch.zeros_like(self.reason_factor_route_enabled)
                    diagnostic_positive = factor_output["factor_positive_evidence"] * 0.0
                    diagnostic_negative = factor_output["factor_negative_evidence"] * 0.0
                    diagnostic_coordinates = factor_output["sampling_coordinates"]
                    diagnostic_features_typed = factor_output["sampled_features"] * 0.0
                    diagnostic_attention_typed = factor_output["sample_attention"] * 0.0
                else:
                    reliability = continuous_route_weight.roll(1, 1)
                    diagnostic_features = factor_output["factor_features"].roll(1, 1) * reliability.unsqueeze(-1)
                    diagnostic_masks = factor_output["factor_soft_masks"].roll(1, 1) * reliability.unsqueeze(-1).unsqueeze(-1)
                    diagnostic_enabled = torch.ones_like(self.reason_factor_route_enabled)
                    diagnostic_positive = factor_output["factor_positive_evidence"].roll(1, 1)
                    diagnostic_negative = factor_output["factor_negative_evidence"].roll(1, 1)
                    diagnostic_coordinates = factor_output["sampling_coordinates"].roll(1, 1)
                    diagnostic_features_typed = factor_output["sampled_features"].roll(1, 1)
                    diagnostic_attention_typed = factor_output["sample_attention"].roll(1, 1)
                diagnostic_latent = self.reason_latent_decoder(
                    self._detached_pyramid(reason_pyramid),
                    diagnostic_features,
                    diagnostic_masks,
                    diagnostic_enabled,
                    diagnostic_positive.detach(),
                    diagnostic_negative.detach(),
                    diagnostic_coordinates,
                    diagnostic_features_typed,
                    diagnostic_attention_typed,
                    target_utility_state["semantic_compatibility"],
                )
                diagnostic_observation = self.observation_model(
                    diagnostic_latent["reason_logits_latent"], factor_output["factor_visibility_prob"], factor_output["factor_uncertainty"]
                )
                diagnostic_observed = self.reason_observed_mixer(
                    reason_visual["reason_visual_observed_logits"], diagnostic_observation["reason_observation_logits"],
                    latent_enabled=latent_enabled, route_mass=diagnostic_latent["reason_route_mass"],
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
