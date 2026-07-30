from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor, nn

from .meter_calalign_foundation import METERCalAlignFoundation
from .meter_reason_decoder import METERPrivateReasonDecoder
from .meter_semantic_action import FactorSpecificActionTransport
from .meter_signed_factors import TypedEvidenceStateHead


def cross_sample_swap_typed_evidence(
    evidence: dict[str, Tensor],
) -> dict[str, Tensor]:
    keys = (
        "factor_typed_token",
        "factor_action_token",
        "factor_action_value_token",
        "factor_state_prob",
        "factor_reliability",
        "factor_observability",
    )
    return {
        key: torch.roll(value, shifts=1, dims=0) if key in keys else value
        for key, value in evidence.items()
    }


class METEROIAModel(nn.Module):
    """TESA formal graph with action-owned factors and a hard reason firewall."""

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
        action_max_visual_rms: float = 5.0,
        action_max_delta: float = 1.0,
        action_logit_norm_cap: float = 20.0,
        **_: Any,
    ) -> None:
        super().__init__()
        self.foundation = METERCalAlignFoundation(
            dim=dim,
            action_dim=action_dim,
            reason_dim=reason_dim,
            selected_layers=selected_layers,
            pretrained_weights=pretrained_weights,
            action_logit_norm_cap=action_logit_norm_cap,
            use_mock_dino=use_mock_dino,
        )
        self.typed_factors = TypedEvidenceStateHead(
            dim=dim,
            factor_dim=reason_dim,
            num_layers=len(selected_layers),
            schema_path=schema_path,
        )
        self.action_transport = FactorSpecificActionTransport(
            dim=dim,
            action_dim=action_dim,
            factor_dim=reason_dim,
            rank=factor_rank,
            correction_fraction=action_correction_fraction,
            max_visual_rms=action_max_visual_rms,
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

        timing: dict[str, float] = {}
        start = stamp()
        base = self.foundation.decode_foundation(field)
        after_foundation = stamp()
        factors = self.typed_factors(
            base["factor_base_nodes"],
            base["patch_tokens_by_layer"],
            progress=progress,
        )
        factor_token = factors["factor_typed_token"]
        factor_action_token = factors["factor_action_token"]
        factor_action_value_token = factors["factor_action_value_token"]
        reliability = factors["factor_reliability"]
        state_prob = factors["factor_state_prob"]
        if "factor_off" in diagnostic_modes:
            reliability = torch.zeros_like(reliability)
        if "state_off" in diagnostic_modes:
            uniform = factors["factor_state_valid_mask"].to(state_prob)
            uniform = uniform.unsqueeze(0).expand(state_prob.shape[0], -1, -1)
            state_prob = uniform / uniform.sum(-1, keepdim=True)
            factor_token = self.typed_factors.compose_typed_token(
                factors["factor_global_token"],
                factors["factor_anchor_token"],
                state_prob,
            )
            factor_action_token = self.typed_factors.compose_action_token(
                factors["factor_anchor_token"], state_prob
            )
        if "schema_corruption" in diagnostic_modes:
            factor_token = torch.roll(factor_token, 1, 1)
            factor_action_token = torch.roll(factor_action_token, 1, 1)
            factor_action_value_token = torch.roll(
                factor_action_value_token, 1, 1
            )
        if "cross_sample_swap" in diagnostic_modes and factor_token.shape[0] > 1:
            swapped = cross_sample_swap_typed_evidence(factors)
            factor_token = swapped["factor_typed_token"]
            factor_action_token = swapped["factor_action_token"]
            factor_action_value_token = swapped["factor_action_value_token"]
            state_prob = swapped["factor_state_prob"]
            reliability = swapped["factor_reliability"]
        if "state_corruption" in diagnostic_modes:
            state_prob = torch.roll(state_prob, 1, -1)
            factor_token = self.typed_factors.compose_typed_token(
                factors["factor_global_token"],
                factors["factor_anchor_token"],
                state_prob,
            )
            factor_action_token = self.typed_factors.compose_action_token(
                factors["factor_anchor_token"], state_prob
            )
        after_factor = stamp()
        action = self.action_transport(
            base["action_logits_calalign"],
            base["action_nodes"],
            factor_action_token,
            reliability,
            factors["factor_action_ownership"],
            factor_value_token=factor_action_value_token,
            factor_source=factors["factor_observability"],
            progress=progress,
            update_running_stats=update_semantic_stats,
        )
        targeted_schema_action = next(
            (
                int(mode.rsplit("_", 1)[-1])
                for mode in diagnostic_modes
                if mode.startswith("schema_target_")
            ),
            None,
        )
        if targeted_schema_action is not None:
            if not 0 <= targeted_schema_action < self.foundation.action_dim:
                raise ValueError("schema_target action index is out of range")
            selected_factor = action["action_factor_weights"][
                :, targeted_schema_action
            ].argmax(-1)
            corrupted_token = factor_action_token.clone()
            corrupted_value_token = factor_action_value_token.clone()
            rolled_token = torch.roll(factor_action_token, 1, 1)
            rolled_value_token = torch.roll(factor_action_value_token, 1, 1)
            batch_index = torch.arange(
                factor_token.shape[0], device=factor_token.device
            )
            corrupted_token[batch_index, selected_factor] = rolled_token[
                batch_index, selected_factor
            ]
            corrupted_value_token[batch_index, selected_factor] = (
                rolled_value_token[batch_index, selected_factor]
            )
            action = self.action_transport(
                base["action_logits_calalign"],
                base["action_nodes"],
                corrupted_token,
                reliability,
                factors["factor_action_ownership"],
                factor_value_token=corrupted_value_token,
                factor_source=factors["factor_observability"],
                progress=progress,
                update_running_stats=False,
            )
        after_action = stamp()
        reason = self.reason_decoder(
            patch_tokens_by_layer=base["patch_tokens_by_layer"],
            reason_logits_calalign=base["reason_logits_calalign"],
            factor_typed_token=factor_token,
            factor_reliability=reliability,
            factor_groundable_mask=factors["factor_groundable_mask"],
            progress=progress,
        )
        if "reason_correction_off" in diagnostic_modes:
            reason["reason_logits_final"] = reason["reason_logits_global"]
        after_reason = stamp()
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
            **action,
            **reason,
            "factor_typed_token": factor_token,
            "factor_action_token": factor_action_token,
            "factor_action_value_token": factor_action_value_token,
            "factor_state_prob": state_prob,
            "factor_state_prob_effective": state_prob,
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

    def forward_mirror_pair(
        self,
        images: Tensor,
        *,
        progress: float = 1.0,
    ) -> dict[str, Any]:
        """Run original and horizontal-mirror images as one paired audit."""
        from fate_oia.losses.meter_grounding_losses import mirror_equivariance_loss

        original = self.forward(images, progress=progress)
        mirrored = self.forward(torch.flip(images, dims=[-1]), progress=progress)
        loss, report = mirror_equivariance_loss(
            original,
            mirrored,
            factor_pairs=self.typed_factors.mirror_pairs,
        )
        return {
            "original": original,
            "mirrored": mirrored,
            "mirror_equivariance_loss": loss,
            "mirror_equivariance": report,
        }
