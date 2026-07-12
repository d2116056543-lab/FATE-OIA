from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .acpr_dino_field import ACPRDinoFieldExtractor
from .mosaic_action_decoder import MOSAICActionDecoder
from .mosaic_native_semantics import load_mosaic_schema_bundle
from .mosaic_observable_predicates import MOSAICObservablePredicateLayer
from .mosaic_reason_decoder import MOSAICReasonDecoder
from .mosaic_state_composer import MOSAICSupportVetoComposer
from .mosaic_visual_pyramid import MOSAICVisualPyramid


class _MOSAICPyramidLaneAdapter(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projections = nn.ModuleDict(
            {
                "F_hi": nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                "F_mid": nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                "F_ctx": nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            }
        )
        for projection in self.projections.values():
            nn.init.zeros_(projection.weight)
            diagonal = torch.arange(dim)
            projection.weight.data[diagonal, diagonal, 0, 0] = 1.0

    def forward(self, pyramid: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: projection(pyramid[name]) for name, projection in self.projections.items()}


class MOSAICADModel(nn.Module):
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
        decoder_layers: int = 2,
        self_attention_heads: int = 4,
        highres_topk: int = 256,
        midres_topk: int = 128,
        anchors_per_factor: int = 2,
        typed_attention_heads: int = 4,
        point_samples: int = 8,
        curve_samples: int = 12,
        region_samples: int = 12,
        spatial_prior_scale_init: float = 0.05,
        spatial_prior_scale_max: float = 0.20,
        spatial_prior_dropout: float = 0.50,
        content_temperature_init: float = 0.07,
        state_residual_cap: float = 0.20,
    ) -> None:
        super().__init__()
        bundle = load_mosaic_schema_bundle(config_root)
        self.schema_bundle = bundle
        self.dino = ACPRDinoFieldExtractor(
            arch=backbone_arch,
            patch_size=backbone_patch_size,
            selected_layers=selected_layers,
            checkpoint_key=checkpoint_key,
            pretrained_weights=pretrained_weights,
            freeze_backbone=True,
            use_mock_dino=use_mock_dino,
        )
        dim = self.dino.dim
        self.visual_pyramid = MOSAICVisualPyramid(input_dim=dim, output_dim=dim)
        self.action_adapter = _MOSAICPyramidLaneAdapter(dim)
        self.reason_adapter = _MOSAICPyramidLaneAdapter(dim)
        self.observable_predicates = MOSAICObservablePredicateLayer(
            bundle["factors"],
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
        factor_names = [factor["name"] for factor in bundle["factors"]]
        factor_index = {name: index for index, name in enumerate(factor_names)}
        contradiction_mask = torch.zeros(len(factor_names), len(factor_names), dtype=torch.bool)
        for factor in bundle["factors"]:
            source = factor_index[factor["name"]]
            for contradicted_name in factor["contradicts"]:
                target = factor_index[contradicted_name]
                contradiction_mask[source, target] = True
                contradiction_mask[target, source] = True
        if not torch.equal(contradiction_mask, contradiction_mask.T) or contradiction_mask.diagonal().any():
            raise ValueError("MOSAIC factor contradiction mask must be symmetric and irreflexive")
        self.register_buffer("factor_contradiction_mask", contradiction_mask, persistent=True)
        self.state_composer = MOSAICSupportVetoComposer(
            factor_names,
            bundle["states"],
            dim=dim,
            state_residual_cap=state_residual_cap,
        )
        self.action_decoder = MOSAICActionDecoder(
            len(bundle["states"]),
            dim=dim,
            decoder_layers=decoder_layers,
            self_attention_heads=self_attention_heads,
            highres_topk=highres_topk,
            midres_topk=midres_topk,
        )
        self.reason_decoder = MOSAICReasonDecoder(
            factor_names,
            bundle["states"],
            bundle["reason_observation"],
            dim=dim,
            decoder_layers=decoder_layers,
            self_attention_heads=self_attention_heads,
            highres_topk=highres_topk,
            midres_topk=midres_topk,
        )
        self.register_buffer("state_residual_scale", torch.zeros(()), persistent=True)
        self.register_buffer("action_state_gate_cap", torch.zeros(()), persistent=True)
        self.register_buffer("reason_state_contribution_cap", torch.zeros(()), persistent=True)
        self._state_residual_scale_value = 0.0
        self._action_state_gate_cap_value = 0.0
        self._reason_state_contribution_cap_value = 0.0

    @torch.no_grad()
    def set_phase_controls(
        self,
        *,
        state_residual_scale: float,
        action_state_gate_cap: float,
        reason_state_contribution_cap: float = 0.0,
    ) -> None:
        if not 0.0 <= state_residual_scale <= 1.0:
            raise ValueError("state_residual_scale must be in [0,1]")
        if not 0.0 <= action_state_gate_cap <= 0.25:
            raise ValueError("action_state_gate_cap must be in [0,0.25]")
        if not 0.0 <= reason_state_contribution_cap <= 0.20:
            raise ValueError("reason_state_contribution_cap must be in [0,0.20]")
        self.state_residual_scale.fill_(float(state_residual_scale))
        self.action_state_gate_cap.fill_(float(action_state_gate_cap))
        self.reason_state_contribution_cap.fill_(float(reason_state_contribution_cap))
        self._state_residual_scale_value = float(state_residual_scale)
        self._action_state_gate_cap_value = float(action_state_gate_cap)
        self._reason_state_contribution_cap_value = float(reason_state_contribution_cap)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        # Checkpoints store these controls as float32 buffers.  Clamp the
        # restored Python values so 0.20000000298 does not violate the
        # documented inclusive upper bound after serialization.
        self._state_residual_scale_value = min(
            1.0, max(0.0, float(self.state_residual_scale.detach().cpu()))
        )
        self._action_state_gate_cap_value = min(
            0.25, max(0.0, float(self.action_state_gate_cap.detach().cpu()))
        )
        self._reason_state_contribution_cap_value = min(
            0.20, max(0.0, float(self.reason_state_contribution_cap.detach().cpu()))
        )

    def forward(
        self,
        images: torch.Tensor,
        *,
        prior_mode: str = "full",
        return_masks: bool = False,
        return_intermediates: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, Any]]:
        field = self.dino(images)
        pyramid = self.visual_pyramid(field["patch_tokens_by_layer"])
        action_pyramid = self.action_adapter(pyramid)
        reason_pyramid = self.reason_adapter(pyramid)
        factors = self.observable_predicates(reason_pyramid, prior_mode=prior_mode)
        states = self.state_composer(
            factors["factor_positive_evidence"],
            factors["factor_negative_evidence"],
            factors["factor_uncertainty"],
            reason_pyramid["F_ctx"],
            residual_scale=self._state_residual_scale_value / self.state_composer.state_residual_cap,
        )
        action = self.action_decoder(
            action_pyramid,
            states["decision_state_prob"],
            states["decision_state_uncertainty"],
            state_gate_cap=self._action_state_gate_cap_value,
        )
        reason = self.reason_decoder(
            reason_pyramid,
            factors["factor_features"],
            factors["factor_soft_masks"],
            states["decision_state_prob"],
            states["decision_state_uncertainty"],
            state_contribution_cap=self._reason_state_contribution_cap_value,
        )
        output: dict[str, torch.Tensor | dict[str, Any]] = {
            **factors,
            **states,
            **action,
            **reason,
            "action_logits_visual": action["action_visual_logits"],
            "action_logits_state": action["action_state_logits"],
        }
        if not return_masks:
            output.pop("factor_soft_masks", None)
            output.pop("sampling_coordinates", None)
            output.pop("reason_factor_masks", None)
        if return_intermediates:
            # Diagnostics only: expose exact decoder inputs without changing
            # the default training/evaluation path.
            output["_diagnostic_intermediates"] = {
                "action_pyramid": action_pyramid,
                "reason_pyramid": reason_pyramid,
            }
        return output
