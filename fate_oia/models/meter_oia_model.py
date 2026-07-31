from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor, nn

from .meter_calalign_foundation import METERCalAlignFoundation
from .meter_meta_adapters import HECASharedPrivateAdapters
from .meter_reason_decoder import METERPrivateReasonDecoder
from .meter_semantic_action import StateConditionedActionCredit
from .meter_signed_factors import TypedEvidenceStateHead


class METEROIAModel(nn.Module):
    """HECA: detached typed measurement with task-private evidence credit."""

    def __init__(
        self,
        dim: int = 384,
        action_dim: int = 4,
        reason_dim: int = 21,
        selected_layers: tuple[int, ...] = (3, 7, 11),
        pretrained_weights: str = "ckp/reference/dino_deitsmall8_pretrain.pth",
        use_mock_dino: bool = False,
        factor_rank: int = 16,
        schema_path: str | None = None,
        action_correction_fraction: float = 0.20,
        action_max_delta: float = 1.0,
        action_logit_norm_cap: float = 20.0,
        action_measurement_grad_scale: float = 0.05,
        **_: Any,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.reason_dim = int(reason_dim)
        self.foundation = METERCalAlignFoundation(
            dim=dim,
            action_dim=action_dim,
            reason_dim=reason_dim,
            selected_layers=selected_layers,
            pretrained_weights=pretrained_weights,
            action_logit_norm_cap=action_logit_norm_cap,
            use_mock_dino=use_mock_dino,
        )
        self.heca_adapters = HECASharedPrivateAdapters(
            dim=dim,
            action_dim=action_dim,
            reason_dim=reason_dim,
            rank=factor_rank,
        )
        self.typed_factors = TypedEvidenceStateHead(
            dim=dim,
            factor_dim=reason_dim,
            num_layers=len(selected_layers),
            schema_path=schema_path,
            action_measurement_grad_scale=action_measurement_grad_scale,
        )
        self.action_transport = StateConditionedActionCredit(
            dim=dim,
            action_dim=action_dim,
            factor_dim=reason_dim,
            rank=factor_rank,
            correction_fraction=action_correction_fraction,
            max_action_delta=action_max_delta,
        )
        self.reason_decoder = METERPrivateReasonDecoder(
            dim=dim, reason_dim=reason_dim, action_dim=action_dim
        )
        self.reason_decoder.initialize_from_foundation(self.foundation)
        self._encode_call_count = 0

    @property
    def signed_factors(self) -> TypedEvidenceStateHead:
        return self.typed_factors

    def encode_images(self, images: Tensor) -> dict[str, Any]:
        self._encode_call_count += 1
        return self.foundation.encode_images(images)

    def _visual_action(self, action_nodes: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.foundation.trunk.action_visual_head(action_nodes).squeeze(-1)
        return self.foundation.trunk._bound_action_logits(raw)

    def _recompute_state_credit(
        self,
        factors: dict[str, Tensor],
        state_prob: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.typed_factors.compose_action_bridge_token(
            factors["factor_anchor_token"],
            state_prob,
            factors["factor_global_token"],
        )

    def decode_from_field(
        self,
        field: dict[str, Any],
        *,
        progress: float = 1.0,
        diagnostic_modes: tuple[str, ...] = (),
        collect_timing: bool = False,
        update_semantic_stats: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        def stamp() -> float:
            if collect_timing and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        start = stamp()
        base = self.foundation.decode_foundation(field)
        adapted = self.heca_adapters(base["label_nodes"])
        action_visual, action_preclip_norm = self._visual_action(
            adapted["action_nodes"]
        )
        after_foundation = stamp()
        factors = self.typed_factors(
            base["factor_base_nodes"],
            base["patch_tokens_by_layer"],
            progress=progress,
        )
        reliability = factors["factor_reliability"]
        state_prob = factors["factor_state_prob"]
        action_bridge = factors["factor_action_bridge_token"]
        state_prob_credit = factors["factor_state_prob_credit"]
        measurement_token = factors["factor_measurement_token"]
        if "factor_off" in diagnostic_modes:
            reliability = torch.zeros_like(reliability)
        if "state_off" in diagnostic_modes or "state_uniform" in diagnostic_modes:
            valid = factors["factor_state_valid_mask"].to(state_prob)
            state_prob = valid.unsqueeze(0).expand_as(state_prob)
            state_prob = state_prob / state_prob.sum(-1, keepdim=True).clamp_min(1)
            action_bridge, state_prob_credit = self._recompute_state_credit(
                factors, state_prob
            )
            measurement_token = self.typed_factors.compose_typed_token(
                factors["factor_global_token"],
                factors["factor_anchor_token"],
                state_prob,
            )
        if "schema_corruption" in diagnostic_modes:
            action_bridge = torch.roll(action_bridge, 1, 1)
            measurement_token = torch.roll(measurement_token, 1, 1)
        if "cross_sample_swap" in diagnostic_modes and action_bridge.shape[0] > 1:
            action_bridge = torch.roll(action_bridge, 1, 0)
            state_prob_credit = torch.roll(state_prob_credit, 1, 0)
            reliability = torch.roll(reliability, 1, 0)
            measurement_token = torch.roll(measurement_token, 1, 0)
        if "state_corruption" in diagnostic_modes:
            state_prob = torch.roll(state_prob, 1, -1)
            action_bridge, state_prob_credit = self._recompute_state_credit(
                factors, state_prob
            )
        after_factor = stamp()
        action = self.action_transport(
            action_visual,
            adapted["action_nodes"],
            action_bridge,
            state_prob_credit,
            reliability,
            factors["factor_action_ownership"],
            progress=progress,
            update_running_stats=update_semantic_stats,
        )
        after_action = stamp()
        reason = self.reason_decoder(
            reason_logits_calalign=base["reason_logits_calalign"],
            reason_nodes=adapted["reason_nodes"],
            factor_measurement_token=measurement_token,
            factor_reliability=reliability,
            factor_groundable_mask=factors["factor_groundable_mask"],
            progress=progress,
        )
        if "reason_correction_off" in diagnostic_modes:
            reason["reason_logits_final"] = reason["reason_logits_global"]
        after_reason = stamp()
        timing = {}
        if collect_timing:
            timing = {
                "foundation_time": after_foundation - start,
                "factor_time": after_factor - after_foundation,
                "action_time": after_action - after_factor,
                "reason_time": after_reason - after_action,
            }
        return {
            **field,
            **base,
            **factors,
            **adapted,
            **action,
            **reason,
            "action_nodes": adapted["action_nodes"],
            "factor_state_prob": state_prob,
            "factor_state_prob_effective": state_prob,
            "factor_action_bridge_token": action_bridge,
            "factor_state_prob_credit": state_prob_credit,
            "factor_measurement_token": measurement_token,
            "action_visual_preclip_norm": action_preclip_norm,
            "reason_logits_pu_private": reason["reason_logits_final"],
            "branch_logits": {
                "action_visual": action["action_logits_visual"],
                "action_final": action["action_logits_final"],
                "reason_global": reason["reason_logits_global"],
                "reason_final": reason["reason_logits_final"],
            },
            "diagnostic_modes": diagnostic_modes,
            "runtime_timing": timing,
        }

    def forward(
        self,
        images: Tensor,
        *,
        progress: float = 1.0,
        diagnostic_modes: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        return self.decode_from_field(
            self.encode_images(images),
            progress=progress,
            diagnostic_modes=diagnostic_modes,
        )

    def forward_view_pair(
        self,
        images: Tensor,
        views: Tensor,
        *,
        progress: float = 1.0,
    ) -> dict[str, Any]:
        """Encode original and paired view together in exactly one DINO call."""
        batch = images.shape[0]
        field = self.encode_images(torch.cat((images, views), dim=0))
        first: dict[str, Any] = {}
        second: dict[str, Any] = {}
        for key, value in field.items():
            if isinstance(value, Tensor) and value.shape[0] == batch * 2:
                first[key], second[key] = value[:batch], value[batch:]
            else:
                first[key] = second[key] = value
        return {
            "original": self.decode_from_field(first, progress=progress),
            "view": self.decode_from_field(second, progress=progress),
        }

    def forward_mirror_pair(
        self, images: Tensor, *, progress: float = 1.0
    ) -> dict[str, Any]:
        pair = self.forward_view_pair(
            images, torch.flip(images, dims=[-1]), progress=progress
        )
        from fate_oia.losses.meter_grounding_losses import mirror_equivariance_loss

        loss, report = mirror_equivariance_loss(
            pair["original"],
            pair["view"],
            factor_pairs=self.typed_factors.mirror_pairs,
        )
        return {
            "original": pair["original"],
            "mirrored": pair["view"],
            "mirror_equivariance_loss": loss,
            "mirror_equivariance": report,
        }
